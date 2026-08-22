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
                
        # To avoid 2^N explosion, we only branch on two logical sell states:
        # A) Hold onto appreciating stocks, sell only what has peaked.
        # B) Liquidate everything profitable right now to maximize immediate compounding capital.
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
                    
            # 3. Buy Phase
            buyable = []
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
                    buyable.append((max_future_p / current_p, max_future_p - current_p, s, current_p))
                    
            buyable.sort(key=lambda x: (x[0], x[1]), reverse=True)
            
            buy_branches = []
            seen_combos = set()
            
            # Heuristic Bounded Knapsack: Check greedy configurations starting from each profitable stock 
            # to prevent permutation timeouts while catching subset edge cases.
            for start_idx in range(len(buyable)):
                c = new_cap
                combo = {}
                for i in range(start_idx, len(buyable)):
                    _, _, s, p = buyable[i]
                    q = min(avail[year].get(s, 0), c // p)
                    if q > 0:
                        combo[s] = q
                        c -= q * p
                for i in range(0, start_idx):
                    _, _, s, p = buyable[i]
                    q = min(avail[year].get(s, 0), c // p)
                    if q > 0:
                        combo[s] = q
                        c -= q * p
                        
                combo_tuple = tuple(sorted(combo.items()))
                if combo_tuple not in seen_combos:
                    seen_combos.add(combo_tuple)
                    buy_branches.append((combo, c))
                    
            empty_combo = tuple()
            if empty_combo not in seen_combos:
                buy_branches.append(({}, new_cap))
                seen_combos.add(empty_combo)
                
            for buy_combo, final_cap in buy_branches:
                final_avail = {y: dict(v) for y, v in avail.items()}
                final_inv = dict(new_inv)
                buy_actions = []
                
                for s, q in buy_combo.items():
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
                    cost = abs(year - target_y)
                    # Scales linearly and guarantees return to 2037
                    if cost <= energy - abs(target_y - 2037):
                        dfs(
                            target_y,
                            energy - cost,
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