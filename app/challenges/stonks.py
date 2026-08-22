import itertools
from typing import Dict, List
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

import heapq
from typing import Dict, List

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

    # Priority Queue state: (-capital, energy_left, current_year, path_history, inventory, availability)
    pq = [(-tc.capital, tc.energy, 2037, [], {}, initial_avail)]
    visited = {}
    best_path = []
    best_profit = -1

    while pq:
        neg_cap, energy, year, path, inv, avail = heapq.heappop(pq)
        cap = -neg_cap

        # Valid exit condition
        if year == 2037 and cap > best_profit:
            best_profit = cap
            best_path = path

        if energy == 0 and year != 2037:
            continue

        # Prune heavily visited, less profitable states
        state_key = (year, energy)
        if state_key in visited and visited[state_key] >= cap:
            continue
        visited[state_key] = cap

        # 1. Greedy Sell Phase
        new_cap = cap
        new_inv = dict(inv)
        sell_actions = []
        for s, q in inv.items():
            if q > 0 and s in prices.get(year, {}):
                current_p = prices[year][s]
                higher_reachable = False
                for target_y in all_years:
                    if target_y == year or s not in prices.get(target_y, {}): continue
                    cost = abs(year - target_y) + abs(target_y - 2037)
                    if cost <= energy and prices[target_y][s] > current_p:
                        higher_reachable = True
                        break
                # Only sell if we have hit the highest reachable future peak
                if not higher_reachable:
                    new_cap += q * current_p
                    new_inv[s] = 0
                    sell_actions.append(f"s-{s}-{q}")

        # 2. Greedy Buy Phase (Fractional Knapsack)
        new_avail = {y: dict(v) for y, v in avail.items()}
        buy_actions = []
        buyable = []
        
        for s, q in avail.get(year, {}).items():
            if q > 0:
                current_p = prices[year].get(s, 0)
                max_future_p = 0
                for target_y in all_years:
                    if target_y == year or s not in prices.get(target_y, {}): continue
                    cost = abs(year - target_y) + abs(target_y - 2037)
                    if cost <= energy and prices[target_y][s] > max_future_p:
                        max_future_p = prices[target_y][s]
                
                if max_future_p > current_p:
                    buyable.append((max_future_p / current_p, s, current_p))

        # Lock in the highest ROI trades first
        buyable.sort(key=lambda x: x[0], reverse=True)
        for roi, s, p in buyable:
            max_q_affordable = new_cap // p
            q_to_buy = min(max_q_affordable, new_avail[year][s])
            if q_to_buy > 0:
                new_cap -= q_to_buy * p
                new_avail[year][s] -= q_to_buy
                new_inv[s] = new_inv.get(s, 0) + q_to_buy
                buy_actions.append(f"b-{s}-{q_to_buy}")

        current_actions = sell_actions + buy_actions
        new_path = path + current_actions

        if year == 2037 and not current_actions and path:
            continue

        # 3. Jump Phase
        for target_y in all_years:
            if target_y != year:
                cost = abs(year - target_y)
                # Ensure we retain enough energy to return home
                if cost <= energy - abs(target_y - 2037):
                    heapq.heappush(pq, (
                        -new_cap,
                        energy - cost,
                        target_y,
                        new_path + [f"j-{year}-{target_y}"],
                        new_inv,
                        new_avail
                    ))

    return best_path

@router.post("/stonks")
def solve_stonks(payload: List[TestCase]) -> List[List[str]]:
    return [solve_test_case(tc) for tc in payload]