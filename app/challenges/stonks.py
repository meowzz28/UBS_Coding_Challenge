import itertools
from typing import Dict, List, Tuple
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["stonks"])

class StockInfo(BaseModel):
    price: int
    qty: int

class TestCase(BaseModel):
    energy: int
    capital: int
    timeline: Dict[str, Dict[str, StockInfo]]

def get_avail_hash(avail: Dict[int, Dict[str, int]]) -> tuple:
    """Safely hash the availability dictionary for memoization."""
    res = []
    for y, stocks in avail.items():
        s_list = [(s, q) for s, q in stocks.items() if q > 0]
        if s_list:
            res.append((y, tuple(sorted(s_list))))
    return tuple(sorted(res))

def solve_test_case(tc: TestCase) -> List[str]:
    prices = {}
    all_years = set([2037])
    
    # Map all available stock prices
    for y_str, stocks in tc.timeline.items():
        y = int(y_str)
        all_years.add(y)
        prices[y] = {s: d.price for s, d in stocks.items()}

    all_years = sorted(list(all_years), reverse=True)
    initial_avail = {y: {s: d.qty for s, d in stocks.items()} for y_str, stocks in tc.timeline.items()}
    for y in all_years:
        if y not in initial_avail:
            initial_avail[y] = {}

    best_profit = -1
    best_path = []
    memo = {}

    def dfs(year: int, energy: int, cap: int, inv: Dict[str, int], avail: Dict[int, Dict[str, int]], path: List[str]):
        nonlocal best_profit, best_path
        
        # 1. State Pruning
        inv_hash = frozenset((s, q) for s, q in inv.items() if q > 0)
        avail_hash = get_avail_hash(avail)
        state_key = (year, energy, inv_hash, avail_hash)
        
        if state_key in memo and memo[state_key] >= cap:
            return
        memo[state_key] = cap

        # 2. Sell Phase
        must_sell = []
        optional_sell = []
        
        for s, q in inv.items():
            if q == 0: continue
            current_p = prices.get(year, {}).get(s, 0)
            if current_p == 0: continue
            
            max_future_p = 0
            for target_y in all_years:
                if target_y == year: continue
                cost = abs(year - target_y) + abs(target_y - 2037)
                if cost <= energy and prices.get(target_y, {}).get(s, 0) > max_future_p:
                    max_future_p = prices[target_y][s]
                    
            if current_p >= max_future_p:
                must_sell.append(s)
            else:
                optional_sell.append(s)
                
        # Branch on sell states
        sell_branches = [must_sell]
        if optional_sell:
            sell_branches.append(must_sell + optional_sell)
            
        sell_branches = [list(x) for x in set(tuple(sorted(b)) for b in sell_branches)]
        
        for sell_subset in sell_branches:
            new_cap = cap
            new_inv = dict(inv)
            sell_actions = []
            
            for s in sell_subset:
                q = new_inv[s]
                if q > 0:
                    p = prices[year][s]
                    new_cap += q * p
                    new_inv[s] = 0
                    sell_actions.append(f"s-{s}-{q}")
                    
            # 3. Buy Phase: Exact DP Knapsack
            buyable_items = []
            for s, q in avail.get(year, {}).items():
                if q == 0: continue
                current_p = prices.get(year, {}).get(s, 0)
                if current_p == 0 or current_p > new_cap: continue
                
                max_future_p = 0
                for target_y in all_years:
                    if target_y == year: continue
                    cost = abs(year - target_y) + abs(target_y - 2037)
                    if cost <= energy and prices.get(target_y, {}).get(s, 0) > max_future_p:
                        max_future_p = prices[target_y][s]
                        
                if max_future_p > current_p:
                    buyable_items.append((s, current_p, max_future_p - current_p, q))
                    
            # Run DP to extract the exact optimal packing bounds
            dp = {0: 0} 
            combo = {0: {}} 
            
            for s, cost, prof, max_q in buyable_items:
                new_dp = dict(dp)
                new_combo = dict(combo)
                for c, p in dp.items():
                    for q_buy in range(1, max_q + 1):
                        nxt_c = c + q_buy * cost
                        if nxt_c <= new_cap:
                            nxt_p = p + q_buy * prof
                            if nxt_c not in new_dp or nxt_p > new_dp[nxt_c]:
                                new_dp[nxt_c] = nxt_p
                                new_combo[nxt_c] = dict(combo.get(c, {}))
                                new_combo[nxt_c][s] = q_buy
                dp = new_dp
                combo = new_combo
                
            # Extract Pareto Frontier
            sorted_costs = sorted(dp.keys())
            pareto_combos = []
            max_p = -1
            for c in sorted_costs:
                if dp[c] > max_p:
                    max_p = dp[c]
                    pareto_combos.append((c, combo[c]))
                    
            # Limit Pareto branches to avoid TLE while catching edge cases
            if len(pareto_combos) > 10:
                step = len(pareto_combos) / 9.0
                sampled = [pareto_combos[int(i * step)] for i in range(9)]
                if pareto_combos[-1] not in sampled:
                    sampled.append(pareto_combos[-1])
                pareto_combos = sampled
                
            for cost, buy_combo in pareto_combos:
                final_cap = new_cap - cost
                final_avail = {y: dict(v) for y, v in avail.items()}
                final_inv = dict(new_inv)
                buy_actions = []
                
                for s in sorted(buy_combo.keys()):
                    q = buy_combo[s]
                    if q > 0:
                        final_avail[year][s] -= q
                        final_inv[s] = final_inv.get(s, 0) + q
                        buy_actions.append(f"b-{s}-{q}")
                    
                current_actions = sell_actions + buy_actions
                
                is_start = (not path and not current_actions and year == 2037)
                if not current_actions and not is_start:
                    if year == 2037 and final_cap > best_profit:
                        best_profit = final_cap
                        best_path = list(path)
                    if year != 2037:
                        continue 
                
                if year == 2037 and final_cap > best_profit:
                    best_profit = final_cap
                    best_path = path + current_actions
                    
                # 4. Jump Phase
                for target_y in all_years:
                    if target_y == year: continue
                    jump_cost = abs(year - target_y)
                    # Scales linearly and guarantees return home safely[cite: 1]
                    if jump_cost <= energy - abs(target_y - 2037):
                        dfs(
                            target_y,
                            energy - jump_cost,
                            final_cap,
                            final_inv,
                            final_avail,
                            path + current_actions + [f"j-{year}-{target_y}"]
                        )

    dfs(2037, tc.energy, tc.capital, {}, initial_avail, [])
    return best_path

@router.post("/stonks")
def solve_stonks(payload: List[TestCase]) -> List[List[str]]:
    return [solve_test_case(tc) for tc in payload]