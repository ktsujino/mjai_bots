import mjx
import sys
import json
from gateway import MjxGateway
import random
from mjx.const import ActionType, TileType, EventType

class RuleBasedAgent(mjx.Agent):
    def __init__(self, enable_opens=True,
                 enable_heuristic_score=True,
                 shanten_aware_opens=True,
                 tanyao_fanpai_aware_opens=True,
                 type_heuristics=True,
                 fanpai_heuristics=True,
                 dora_heuristics=True,
                 adjacency_heuristics=True,
                 betaori_heuristics=True,
                 verbose=False) -> None:
        super().__init__()
        self.enable_opens = enable_opens
        self.enable_heuristic_score = enable_heuristic_score
        self.shanten_aware_opens = shanten_aware_opens
        self.tanyao_fanpai_aware_opens = tanyao_fanpai_aware_opens
        self.type_heuristics = type_heuristics
        self.fanpai_heuristics = fanpai_heuristics
        self.dora_heuristics = dora_heuristics
        self.adjacency_heuristics = adjacency_heuristics
        self.betaori_heuristics = betaori_heuristics
        self.verbose = verbose
 
    def _fanpais(self, observation):
        fanpais = [TileType.WD, TileType.GD, TileType.RD]
        fanpais.append(TileType(int(TileType.EW) + observation.round() // 4))
        fanpais.append(TileType(int(TileType.EW) + (int(observation.who()) - int(observation.dealer()) + 4)% 4))
        return fanpais

    def _is_open_tanyao_or_fanpai(self, observation, open):
        tiles = open.tiles()
        fanpais = self._fanpais(observation)
        ok = None
        if all([tile.num() is not None and tile.num() >= 2 and tile.num() <= 8 for tile in tiles]) or all([tile.type() in fanpais for tile in tiles]):
            ok = True
        else:
            ok = False
        if self.verbose:
            print('open:  ',
                [int(tile.type()) for tile in open.tiles()],
                ok
            )
        return ok
               
    def _shanten_and_effective_tiles_after_discard(self, curr_hand, tile_to_discard):
        curr_hand_dict = json.loads(curr_hand.to_json())
        assert tile_to_discard.id() in curr_hand_dict['closedTiles']
        curr_hand_dict['closedTiles'].remove(tile_to_discard.id())
        hand_after_discard = mjx.Hand(json.dumps(curr_hand_dict))
        shanten_number = hand_after_discard.shanten_number()
        effective_draw_types = hand_after_discard.effective_draw_types()
        return shanten_number, len(effective_draw_types)

    def _is_shanten_reduced_by_open(self, curr_hand, action):
        assert action.type() in [ActionType.CHI, ActionType.PON, ActionType.OPEN_KAN]
        curr_shanten = curr_hand.shanten_number()
        ok = False
        hand_after_open = curr_hand
        for tile in curr_hand.closed_tiles():
            if tile.type() in [tile.type() for tile in action.open().tiles()]:
                continue
            curr_hand_dict = json.loads(curr_hand.to_json())
            curr_hand_dict['closedTiles'].append(action.open().stolen_tile().id())
            curr_hand_dict['closedTiles'].remove(tile.id())
            hand_after_open = mjx.Hand(json.dumps(curr_hand_dict))
            shanten_after_open = hand_after_open.shanten_number()
            if shanten_after_open < curr_shanten:
                ok = True
                break
        if self.verbose:
            print('open:  ',
                  [int(tile.type()) for tile in curr_hand.closed_tiles()],
                  [int(tile.type()) for tile in hand_after_open.closed_tiles()],
                  curr_shanten,
                  shanten_after_open,
                  ok)     
        return ok

    def _adjacency_heuristic_score(self, curr_hand, tile):
        closed_tile_types = curr_hand.closed_tile_types()
        def is_single(tile):
            return closed_tile_types[tile.type()] == 1
        def is_head(tile):
            return closed_tile_types[tile.type()] >= 2
        def is_kotsu(tile):
            return closed_tile_types[tile.type()] >= 3
        def has_relative(tile, offset):
            if offset == 0:
                return True
            to_find = int(tile.type()) + offset
            if to_find < TileType.M1 or to_find > TileType.S9 or int(tile.type()) // 9 != int(to_find) // 9:
                return False
            return closed_tile_types[to_find] > 0
        def is_shuntsu(tile):
            for start in range(-2, 1):
                if has_relative(tile, start) and has_relative(tile, start+1) and has_relative(tile, start+2):
                    return True
            return False
        def has_neighbors(tile):
            return has_relative(tile, -1) or has_relative(tile, 1)
        def has_skip_neighbors(tile):
            return has_relative(tile, -2) or has_relative(tile, 2)
        def is_penchan(tile):
            return not is_shuntsu(tile) and (
                tile.num() == 1 and has_relative(tile, 1) or
                tile.num() == 2 and has_relative(tile, -1) or
                tile.num() == 8 and has_relative(tile, 1) or
                tile.num() == 9 and has_relative(tile, -1))
        def is_ryanmen(tile):
            return not is_shuntsu(tile) and not is_penchan(tile) and has_neighbors(tile)
        def is_ryankan(tile):
            return not is_shuntsu(tile) and not has_neighbors(tile) and (
                has_relative(tile, -4) and has_relative(tile, -2) or
                has_relative(tile, -2) and has_relative(tile, 2) or
                has_relative(tile, 2) and has_relative(tile, 4))
        score = 0
        if is_head(tile):
            score += 0.2
        elif is_ryanmen(tile):
            score += 0.3
        elif has_skip_neighbors(tile):
            score += 0.2
        elif is_penchan(tile):
            score += 0.1
        return score

    def _get_riichi_players(self, observation):
        riichi_players = [False] * 4
        for i in range(4):
            if any([e.type() == EventType.RIICHI and e.who() == i for e in observation.events()]):
                riichi_players[i] = True
        return riichi_players

    def _under_riichi(self, observation):
        return any(self._get_riichi_players(observation))

    def _safe_tiles(self, observation):
        discarded = [set() for _ in range(4)]
        safe = [set() for _ in range(4)]
        riichi_players = [False] * 4

        for e in observation.events():
            if e.type() != EventType.DISCARD and e.type() != EventType.TSUMOGIRI:
                continue
            discarded[e.who()].add(e.tile().type())
            for player in range(4):
                safe[player].add(e.tile().type())
            if e.type() != EventType.TSUMOGIRI:
                safe[e.who()] = set()
        for player in range(4):
            for t in discarded[player]:
                safe[player].add(t)    
        return safe

    def _betaori_score(self, observation, tile):
        score = 0
        riichi_players = self._get_riichi_players(observation)
        if not any(riichi_players):
            return score
        safe_tiles = self._safe_tiles(observation)
        for player in range(4):
            if not riichi_players[player]:
                continue
            if tile.type() in safe_tiles[player]:
                score += 20000
        score -= 10000 * observation.doras().count(tile.type())
        if tile.is_red():
            score -= 10000
        num = tile.num()
        if num == None:
            score += 200
        if num == 1 or num == 9:
            score += 100
        return score
            
    def _heuristic_score(self, observation, curr_hand, action):
        shanten, effective_tiles = self._shanten_and_effective_tiles_after_discard(curr_hand, action.tile())
              
        score = 13000
        score -= 1000 * shanten
        score += 10 * effective_tiles
        num = action.tile().num()
        fanpais = self._fanpais(observation)
        if self.type_heuristics:
            if num == None:
                score += 6
            if num == 1 or num == 9:
                score += 4
            if num == 2 or num == 8:
                score += 2
        if self.fanpai_heuristics:
            if action.tile().type() in fanpais:
                score -= 1
        if self.dora_heuristics:
            score -= 2 * observation.doras().count(action.tile().type())
            if action.tile().is_red():
                score -= 2
        if self.adjacency_heuristics:
            score -= self._adjacency_heuristic_score(curr_hand, action.tile())
        if self.betaori_heuristics:
            score += self._betaori_score(observation, action.tile())
        if self.verbose:
            print('score: ',
                [int(tile.type()) for tile in curr_hand.closed_tiles()],
                [[int(tile.type()) for tile in open.tiles()] for open in observation.curr_hand().opens()],
                action.type(),
                int(action.tile().type()) if action.tile() else None,
                shanten,
                effective_tiles,
                score,
            )
        return score


    def act(self, observation: mjx.Observation) -> mjx.Action:
        action = self._act(observation)
        if self.verbose:
            print('acted: ',
              [int(tile.type()) for tile in observation.curr_hand().closed_tiles()],
              [[int(tile.type()) for tile in open.tiles()] for open in observation.curr_hand().opens()],
              action.type(),
              int(action.tile().type()) if action.tile() else '',
              [int(tile.type()) for tile in action.open().tiles()] if action.open() else '',
            )
        return action

    def _act(self, observation: mjx.Observation) -> mjx.Action:
        curr_hand = observation.curr_hand()
        legal_actions = observation.legal_actions()
        if len(legal_actions) == 1:
            return legal_actions[0]

        # if possible, just win
        win_actions = [a for a in legal_actions if a.type() in [ActionType.TSUMO, ActionType.RON]]
        if len(win_actions) >= 1:
            assert len(win_actions) == 1
            return win_actions[0]

        # if possible, just riichi
        riichi_actions = [a for a in legal_actions if a.type() == ActionType.RIICHI]
        if len(riichi_actions) >= 1:
            assert len(riichi_actions) == 1
            return riichi_actions[0]

        # if possible, abort
        abort_actions = [a for a in legal_actions if a.type() == ActionType.ABORTIVE_DRAW_NINE_TERMINALS]
        if len(abort_actions) >= 1:
            return abort_actions[0]

        pass_actions = [a for a in legal_actions if a.type() == ActionType.PASS]

        # pon/chi/open kan
        steal_actions = [a for a in legal_actions if a.type() in [ActionType.CHI, ActionType.PON, ActionType, ActionType.OPEN_KAN]]
        if self.enable_opens and len(steal_actions) >= 1:
            if self.betaori_heuristics and self._under_riichi(observation) and len(pass_actions) > 0:
                return pass_actions[0]
            if curr_hand.shanten_number() == 0 and len(pass_actions) > 0:
                return pass_actions[0]
            if self.shanten_aware_opens or self.tanyao_fanpai_aware_opens:
                for action in steal_actions:
                    good = True
                    if self.shanten_aware_opens and not self._is_shanten_reduced_by_open(curr_hand, action):
                        good = False
                    if self.tanyao_fanpai_aware_opens and not self._is_open_tanyao_or_fanpai(observation, action.open()):
                        good = False
                    if good:
                        return action
                if len(pass_actions) > 0:
                    return pass_actions[0]
            else:
                return random.choice(legal_actions)

        # closed kan/added kan
        kan_actions = [a for a in legal_actions if a.type() in [ActionType.CLOSED_KAN, ActionType.ADDED_KAN]]
        if self.enable_opens and len(kan_actions) >= 1 and not (self.betaori_heuristics and self._under_riichi(observation)):
            if curr_hand.shanten_number() == 0:
                return random.choice(kan_actions)

        # discard/tsumogiri
        legal_discards = [a for a in legal_actions if a.type() in [ActionType.DISCARD, ActionType.TSUMOGIRI]]
        if not legal_discards:
            return random.choice(legal_actions)
        if self.enable_heuristic_score:
            legal_discards.sort(key=lambda action:self._heuristic_score(observation, curr_hand, action))
            return legal_discards[-1]  # one with highest heuristic score
        else:
            # minimize shanten
            effective_discard_types = observation.curr_hand().effective_discard_types()
            effective_discards = [a for a in legal_discards if a.tile().type() in effective_discard_types]
            if len(effective_discards) > 0:
                return random.choice(effective_discards)
        sys.stdout.flush()

def main():
    player_id = int(sys.argv[1])
    assert player_id in range(4)
    bot = MjxGateway(player_id, RuleBasedAgent())

    while True:
        line = sys.stdin.readline().strip()
        resp = bot.react(line)
        sys.stdout.write(resp + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
