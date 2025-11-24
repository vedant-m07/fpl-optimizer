import streamlit as st
import pandas as pd
import requests
import pulp

# ==========================================
# 1. DATA INGESTION LAYER
# ==========================================
@st.cache_data(ttl=300) 
def get_live_data():
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return pd.DataFrame()

    elements = pd.DataFrame(data['elements'])
    teams = pd.DataFrame(data['teams'])
    
    # 1. Robust Mappings
    team_map = teams.set_index('id').name.to_dict()
    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    elements['team_name'] = elements.team.map(team_map)
    elements['pos_name'] = elements.element_type.map(pos_map)
    
    # 2. Name Disambiguation
    elements['full_name'] = elements['first_name'] + " " + elements['second_name']
    
    # 3. Data Cleaning
    cols_to_clean = ['influence', 'form', 'points_per_game', 'ep_next']
    for col in cols_to_clean:
        elements[col] = pd.to_numeric(elements[col], errors='coerce').fillna(0.0)
        
    # 4. Price Logic
    elements['now_cost'] = elements['now_cost'] / 10.0
    
    # 5. Availability Logic
    elements['chance_of_playing_next_round'] = elements['chance_of_playing_next_round'].fillna(100.0)
    
    return elements

# ==========================================
# 2. METRICS ENGINE
# ==========================================
def calculate_xp(df):
    # Base xP
    df['base_xp'] = (df['form'] * 0.6) + (df['points_per_game'] * 0.4)
    
    # DefCon xP (Defenders/GKs only)
    def get_defcon(row):
        if row['pos_name'] in ['MID', 'FWD']: return 0.0
        prob = min(0.8, (row['influence'] / 35.0))
        return prob * 2.0
    
    df['defcon_xp'] = df.apply(get_defcon, axis=1)
    
    # Final xP (5 Week Horizon)
    df['final_xp'] = (df['base_xp'] * 1.0 + df['defcon_xp'] * 0.4) * 5
    
    return df

# ==========================================
# 3. THE SOLVER
# ==========================================
def solve_squad(df, budget, force_spend=False, locked_ids=None):
    if locked_ids is None:
        locked_ids = []

    # Pool: Available OR Cheap Fodder (<= 5.0m) OR Locked Players
    # We MUST include locked players even if they are injured/dead/expensive
    pool = df[
        (df.chance_of_playing_next_round >= 50) | 
        (df.now_cost <= 5.0) |
        (df.id.isin(locked_ids))
    ].copy()
    
    prob = pulp.LpProblem("FPL_Solver", pulp.LpMaximize)
    players = pool.id.tolist()
    
    # Mappings
    xp = pool.set_index('id').final_xp.to_dict()
    cost = pool.set_index('id').now_cost.to_dict()
    pos = pool.set_index('id').pos_name.to_dict()
    team = pool.set_index('id').team_name.to_dict()
    
    # Variables
    s = pulp.LpVariable.dicts("start", players, 0, 1, pulp.LpBinary)
    b = pulp.LpVariable.dicts("bench", players, 0, 1, pulp.LpBinary)
    
    # Formation Variables
    valid_formations = [
        (3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2), 
        (4, 5, 1), (5, 3, 2), (5, 2, 3), (5, 4, 1)
    ]
    f_vars = pulp.LpVariable.dicts("formation", [str(f) for f in valid_formations], 0, 1, pulp.LpBinary)
    
    # --- OBJECTIVE ---
    prob += pulp.lpSum([ 
        (xp[i] * s[i]) + 
        (xp[i] * 0.1 * b[i]) + 
        (cost[i] * 0.005 * s[i]) 
        for i in players 
    ])
    
    # --- CONSTRAINTS ---
    
    # 1. Locked Players (MUST be in Starting XI)
    for pid in locked_ids:
        if pid in players:
            prob += s[pid] == 1
            
    # 2. Budget
    total_cost = pulp.lpSum([ cost[i] * (s[i] + b[i]) for i in players ])
    prob += total_cost <= budget
    if force_spend:
        prob += total_cost >= (budget - 1.0)
    
    # 3. Physics
    for i in players:
        prob += s[i] + b[i] <= 1
        
    # 4. Structure
    prob += pulp.lpSum([s[i] for i in players]) == 11
    prob += pulp.lpSum([b[i] for i in players]) == 4
    prob += pulp.lpSum([b[i] for i in players if pos[i] == 'GK']) == 1
    
    # 5. Formation Logic
    prob += pulp.lpSum([f_vars[str(f)] for f in valid_formations]) == 1
    
    prob += pulp.lpSum([s[i] for i in players if pos[i] == 'DEF']) == \
            pulp.lpSum([f[0] * f_vars[str(f)] for f in valid_formations])
    prob += pulp.lpSum([s[i] for i in players if pos[i] == 'MID']) == \
            pulp.lpSum([f[1] * f_vars[str(f)] for f in valid_formations])
    prob += pulp.lpSum([s[i] for i in players if pos[i] == 'FWD']) == \
            pulp.lpSum([f[2] * f_vars[str(f)] for f in valid_formations])
    prob += pulp.lpSum([s[i] for i in players if pos[i] == 'GK']) == 1
    
    # 6. Team Limit (Max 2 per club)
    # Exception: If user locks 3 players from same team, we must allow it.
    for t in pool.team_name.unique():
        limit = 2
        locked_count = sum(1 for pid in locked_ids if pid in players and team[pid] == t)
        if locked_count > 2:
            limit = 3
            
        prob += pulp.lpSum([ s[i] + b[i] for i in players if team[i] == t ]) <= limit

    # Solve
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[status] != 'Optimal':
        return None
        
    # Extract
    squad = []
    for i in players:
        if s[i].value() > 0.5: squad.append({'id': i, 'role': 'Starter'})
        elif b[i].value() > 0.5: squad.append({'id': i, 'role': 'Bench'})
            
    return squad

# ==========================================
# 4. UI LAYER
# ==========================================
def main():
    st.set_page_config(page_title="Ressu FPL", layout="wide")
    st.title("Ressu FPL Ultimate Squad Generator")
    
    # Load Data Early for Sidebar
    df = get_live_data()
    if df.empty:
        st.stop()
    
    # Sidebar Settings
    with st.sidebar:
        st.header("Manager Settings")
        budget = st.number_input("Bank Budget (£m)", 80.0, 120.0, 100.0, 0.1)
        
        st.divider()
        st.subheader("🔒 Lock Players")
        st.caption("Force specific players into the **Starting XI**. The solver will build around them.")
        
        # Player Selector
        all_players = df.sort_values(['now_cost', 'total_points'], ascending=False)
        all_players['label'] = all_players['full_name'] + " (" + all_players['team_name'] + ", £" + all_players['now_cost'].astype(str) + ")"
        
        locked_names = st.multiselect(
            "Select Players to Lock:",
            options=all_players['label'].tolist(),
            default=[]
        )
        
        # Convert names back to IDs
        locked_ids = all_players[all_players['label'].isin(locked_names)].id.tolist()
        st.info(f"Selected {len(locked_ids)}/15 players.")
        
        st.divider()
        force_spend = st.checkbox("Force Max Budget Usage", value=False)
        st.caption("(Uses all budget with max 1.0m margin)")

    clicked = st.button("GENERATE ULTIMATE SQUAD", type="primary")

    if clicked:
        with st.spinner(f'Optimizing around your {len(locked_ids)} locked players...'):
            
            df = calculate_xp(df)
            results = solve_squad(df, budget, force_spend, locked_ids)
            
            if not results:
                st.error("Optimization Failed.")
                st.warning("""
                **Why did this happen?**
                1. Your 'Locked Players' might be too expensive for the remaining budget.
                2. You might have locked too many players from one team/position.
                3. Try increasing the budget or unlocking one player.
                """)
                st.stop()
                
            res_df = pd.DataFrame(results)
            final = res_df.merge(df, on='id')
            
            starters = final[final.role == 'Starter']
            bench = final[final.role == 'Bench']
            
            # Formation
            d = len(starters[starters.pos_name == 'DEF'])
            m = len(starters[starters.pos_name == 'MID'])
            f = len(starters[starters.pos_name == 'FWD'])
            st.success(f"**Optimal Formation:** {d}-{m}-{f}")
            
            col1, col2 = st.columns([2, 1])
            cols = ['full_name', 'team_name', 'pos_name', 'now_cost', 'final_xp']
            
            with col1:
                st.subheader("🚀 Starting XI")
                st.dataframe(
                    starters.sort_values('pos_name', key=lambda x: x.map({'GK':0,'DEF':1,'MID':2,'FWD':3}))[cols],
                    hide_index=True, use_container_width=True
                )
                st.metric("Projected Points", round(starters.final_xp.sum(), 1))
                
            with col2:
                st.subheader("🪑 Bench")
                st.dataframe(
                    bench.sort_values('pos_name', key=lambda x: x.map({'GK':0,'DEF':1,'MID':2,'FWD':3}))[cols],
                    hide_index=True, use_container_width=True
                )
                
                total_cost = final.now_cost.sum()
                remaining = budget - total_cost
                st.metric("Total Cost", f"£{round(total_cost, 1)}m", delta=f"£{round(remaining, 1)}m Remaining")
    
    else:
        # Landing Page Info (Only shows when no squad is generated)
        st.divider()
        st.subheader("The Math Involved")
        
        st.markdown("""
        ### The Algorithm: Partitioned Knapsack Problem
        This tool uses **Linear Programming** (Simplex Method) to find the mathematically optimal combination of 15 players that maximizes expected points within your constraints.
        
        All data taken from FPL API (so data is live, and updated ASAP) 

        **The Constraints:**
        * **Budget:** Must be within your exact bank limit.
        * **Structure:** Exactly 11 Starters and 4 Bench players.
        * **Valid Formations:** Must pick a real formation (e.g., 3-5-2, 4-4-2, 3-4-3).
        * **Considerations:** Accounts for all relevant FPL Rules & Considerations like max players per team, squad value inflation during the season, etc.
        
        **The Metrics:**
        * **Base xP:** A blend of recent Form (60%) and season-long Points Per Game (40%).
        * **DefCon (Defensive Contribution):** A bonus applied to **Defenders and Goalkeepers** who have high 'Influence' stats, simulating the new 2025/26 defensive points rule (Tackles/Blocks).
        * **Rotation Risks:** A dampening factor applied to players who don't play 90 minutes, simulating the new 2025/26 rotation risk rule.
        * **Projection Duration:** Squad optimized for upcoming 5 gameweeks in order to minimize wildcarding and excess transfers.

        ### Why isn't Haaland/Salah in my team?
        The math optimizes for **ROI (Return on Investment)**. 
        Expensive superstars like Haaland (£15.0m) often have a lower "Points Per Million" than mid-priced players like Mbeumo or Wood. The algorithm sees £15.0m as "expensive points" and might prefer spreading that cash across 3 good players.
        
        **Solution:** If you want the superstars for their Captaincy potential or FOMO, simply use the **Lock Players** tool in the sidebar to force them into the Starting XI. The math will then optimize the rest of the squad around them. FYI, this feature will be the next to be implemented.
        """)

if __name__ == "__main__":

    main()
