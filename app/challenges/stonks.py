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

def solve_test_case(tc: TestCase) -> List[str]:
    prices = {}
    initial_avail = {}
    all_years = []
    
    # Parse the timeline
    for y_str, stocks in tc.timeline.items():
        y = int(y_str)
        if y not in all_years:
            all_years.append(y)
            prices[y] = {}
            initial_avail[y] = {}
        for s_name, s_data in stocks.items():
            prices[y][s_name] = s_data.price
            initial_avail[y][s_name] = s_data.qty

    # Ensure 2037 is always present since we must start and return there
    if 2037 not in all_years:
        all_years.append(2037)
        prices[2037] = {}
        initial_avail[2037] = {}

    all_years.sort(reverse=True)
    best_profit = -1
    best_path = []
    memo = {}

    def get_avail_hash(avail: Dict[int, Dict[str, int]]) -> tuple:
        return tuple(sorted((y, tuple(sorted(v.items()))) for y, v in avail.items()))

    def dfs(current_year: int, energy_left: int, capital: int, inventory: Dict[str, int], avail: Dict[int, Dict[str, int]], path_so_far: List[str]):
        nonlocal best_profit, best_path

        state_key = (current_year, energy_left, tuple(sorted(inventory.items())), get_avail_hash(avail))
        if state_key in memo and memo[state_key] >= capital:
            return
        memo[state_key] = capital

        if current_year == 2037:
            if capital > best_profit:
                best_profit = capital
                best_path = path_so_far

        # Determine sell actions
        held_stocks = [s for s, q in inventory.items() if q > 0]
        must_sell, optional_sell = [], []
        for s in held_stocks:
            current_p = prices[current_year].get(s, 0)
            max_future_p = 0
            for y in all_years:
                if y != current_year and s in prices.get(y, {}):
                    req_energy = abs(current_year - y) + abs(y - 2037)
                    if req_energy <= energy_left and prices[y][s] > max_future_p:
                        max_future_p = prices[y][s]
            
            if current_p >= max_future_p and current_p > 0:
                must_sell.append(s)
            elif current_p > 0:
                optional_sell.append(s)

        sell_subsets = []
        for L in range(len(optional_sell) + 1):
            for subset in itertools.combinations(optional_sell, L):
                sell_subsets.append(must_sell + list(subset))

        # Explore all valid sell and buy combinations
        for sell_subset in sell_subsets:
            new_cap = capital
            new_inv = dict(inventory)
            sell_actions = []
            valid_sell = True
            
            for s in sell_subset:
                if s not in prices.get(current_year, {}):
                    valid_sell = False
                    break
                q = new_inv[s]
                new_cap += q * prices[current_year][s]
                new_inv[s] = 0
                sell_actions.append(f"s-{s}-{q}")

            if not valid_sell:
                continue

            available_here = avail[current_year]
            buyable_stocks = []
            for s, q in available_here.items():
                if q == 0: continue
                current_p = prices.get(current_year, {}).get(s, 0)
                if current_p == 0 or current_p > new_cap: continue

                max_future_p = 0
                for y in all_years:
                    if y != current_year and s in prices.get(y, {}):
                        req_energy = abs(current_year - y) + abs(y - 2037)
                        if req_energy <= energy_left and prices[y][s] > max_future_p:
                            max_future_p = prices[y][s]
                
                if max_future_p > current_p:
                    buyable_stocks.append((s, max_future_p / current_p))

            # Prioritize highest ROI to prevent combinatorial explosion
            buyable_stocks.sort(key=lambda x: x[1], reverse=True)
            top_buyable = [x[0] for x in buyable_stocks[:5]]

            combos = set([tuple()])
            for perm in itertools.permutations(top_buyable):
                for bL in range(1, len(perm) + 1):
                    cap = new_cap
                    combo = {}
                    for s in perm[:bL]:
                        p = prices[current_year][s]
                        q = min(avail[current_year][s], cap // p)
                        if q > 0:
                            combo[s] = q
                            cap -= q * p
                    if combo:
                        combos.add(tuple(sorted(combo.items())))

            for buy_combo in combos:
                final_cap = new_cap
                final_avail = {y: dict(v) for y, v in avail.items()}
                final_inv = dict(new_inv)
                buy_actions = []

                for s, q in buy_combo:
                    final_cap -= q * prices[current_year][s]
                    final_avail[current_year][s] -= q
                    final_inv[s] = final_inv.get(s, 0) + q
                    buy_actions.append(f"b-{s}-{q}")

                current_actions = sell_actions + buy_actions

                if current_year == 2037 and energy_left == 0:
                    if final_cap > best_profit:
                        best_profit = final_cap
                        best_path = path_so_far + current_actions
                    continue

                is_start = (not path_so_far and not current_actions and current_year == 2037)
                if not current_actions and not is_start:
                    if current_year == 2037 and final_cap > best_profit:
                        best_profit = final_cap
                        best_path = path_so_far
                    continue

                if current_year == 2037 and final_cap > best_profit:
                    best_profit = final_cap
                    best_path = path_so_far + current_actions

                # Jump generation
                for target_year in all_years:
                    if target_year != current_year:
                        cost = abs(current_year - target_year)
                        if cost <= energy_left - abs(target_year - 2037):
                            dfs(
                                target_year,
                                energy_left - cost,
                                final_cap,
                                final_inv,
                                final_avail,
                                path_so_far + current_actions + [f"j-{current_year}-{target_year}"]
                            )

    dfs(2037, tc.energy, tc.capital, {}, initial_avail, [])
    return best_path

@router.post("/stonks")
def solve_stonks(payload: List[TestCase]) -> List[List[str]]:
    return [solve_test_case(tc) for tc in payload]