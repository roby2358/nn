"""Synthesize a policy from exact game rules; no gradient updates or training."""
from pathlib import Path
import sys
from itertools import combinations
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tictactoe' / 'src'))
from tictactoe.board import Board, LINES
from tictactoe.ai.minimax import value, best_moves


def states():
    seen = {}
    def visit(b):
        key = tuple(b.cells)
        if key in seen or b.game_over:
            return
        seen[key] = b
        for a, c in enumerate(b.cells):
            if c is None:
                child = Board()
                child.cells = b.cells.copy()
                child.turn = b.turn
                child.play(a)
                visit(child)
    visit(Board())
    return list(seen.values())


def tactical(b):
    for who in (b.turn, 'O' if b.turn == 'X' else 'X'):
        moves = set()
        for line in LINES:
            if sum(b.cells[i] == who for i in line) == 2:
                moves.update(i for i in line if b.cells[i] is None)
        if moves:
            return moves
    return set()


def build(extra=()):
    reachable = set(extra)
    def visit(b, me):
        if b.game_over:
            return
        if b.turn == me:
            reachable.add(tuple(b.cells))
            opts = best_moves(b)
            moves = [next(i for i in (4, 0, 2, 6, 8, 1, 3, 5, 7) if i in opts)]
        else:
            moves = [i for i,c in enumerate(b.cells) if c is None]
        for a in moves:
            child = Board(); child.cells = b.cells.copy(); child.turn = b.turn
            child.play(a); visit(child, me)
    for me in 'XO':
        visit(Board(), me)
    boards = [b for b in states() if tuple(b.cells) in reachable and not tactical(b)
              and (1 if b.turn == 'X' else -1) * value(tuple(b.cells), b.turn) >= 0]
    # Each conjunction contains only occupied-square literals and the side to move.
    # Enumerate matching positions. Build an ordered rule cover: a rule must
    # endorse a nonlosing move in every still-uncovered matching position.
    candidates = {}
    safe = []
    for b in boards:
        allowed = set()
        cells = tuple(b.cells)
        side = 1 if b.turn == 'X' else -1
        nxt = 'O' if b.turn == 'X' else 'X'
        for a, c in enumerate(cells):
            if c is not None or side * value(cells[:a] + (b.turn,) + cells[a+1:], nxt) >= 0:
                allowed.add(a)
        safe.append(allowed)
    for j, b in enumerate(boards):
        occupied = [(i, 1 if c == 'X' else -1) for i, c in enumerate(b.cells) if c]
        for bits in range(1 << len(occupied)):
            pattern = tuple(p for k, p in enumerate(occupied) if bits & (1 << k))
            key = (b.turn, pattern)
            if key not in candidates:
                candidates[key] = [set(safe[j]), set()]
            candidates[key][0].intersection_update(safe[j])
            candidates[key][1].add(j)
    uncovered = set(range(len(boards)))
    rules = []
    while uncovered:
        options = []
        for key, (_, matches) in candidates.items():
            actions = set(range(9))
            for j in matches & uncovered:
                actions.intersection_update(safe[j])
            for action in sorted(actions):
                covered = {j for j in matches if boards[j].cells[action] is None}
                options.append((len(covered & uncovered), -len(key[1]), key, action, covered))
        count, _, key, action, matches = max(options, key=lambda o: (o[0], o[1], o[2], -o[3]))
        if not count:
            raise RuntimeError(f'Cannot cover {len(uncovered)} positions')
        rules.append((key, action))
        uncovered -= matches
    print(f'{len(boards)} quiet positions covered by {len(rules)} rules', flush=True)
    W1 = np.zeros((10, 64), np.float32)
    b1 = np.zeros(64, np.float32)
    W2 = np.zeros((64, 9), np.float32)
    b2 = np.zeros(9, np.float32)
    features = []
    # A line sum of two marks identifies an immediate win/block. The turn
    # literal makes a winning line stronger than a blocking line. All cells
    # in that line receive credit; choose() skips occupied ones.
    for mark in 'XO':
        for line in LINES:
            coeff = np.zeros(10)
            coeff[list(line)] = .5 * (1 if mark == 'X' else -1)
            coeff[9] = .1 * (1 if mark == 'X' else -1)
            out = np.zeros(9)
            out[list(line)] = .2
            features.append((coeff, -1., out))
    for rank, ((turn, pattern), action) in enumerate(rules):
        coeff = np.zeros(10)
        for i, sign in pattern:
            coeff[i] = sign
        coeff[9] = 1 if turn == 'X' else -1
        out = np.zeros(9)
        out[action] = .001 * .6 ** rank
        features.append((coeff, -len(pattern) - .5, out))
    if len(features) > 64:
        raise RuntimeError(f'Need {len(features)} hidden neurons')
    for h, (coeff, bias, out) in enumerate(features):
        W1[:, h] = 20 * coeff
        b1[h] = 20 * bias
        W2[h] = out / 2
        b2 += out / 2
    target = Path(__file__).with_name('weights.npz')
    np.savez(target, W1=W1, b1=b1, W2=W2, b2=b2)
    print(f'Saved {target} ({len(features)} active neurons)', flush=True)
    return target


def check(path):
    from tictactoe.ai.policy import PolicyNet
    net = PolicyNet.load(path)
    reached = set()
    losses = 0
    def visit(b, me):
        nonlocal losses
        if b.game_over:
            losses += bool(b.winner and b.winner != me)
            return
        if b.turn == me:
            reached.add(tuple(b.cells))
            moves = [net.choose(b)]
        else:
            moves = [i for i,c in enumerate(b.cells) if c is None]
        for a in moves:
            child = Board(); child.cells = b.cells.copy(); child.turn = b.turn
            child.play(a); visit(child, me)
    for me in 'XO':
        visit(Board(), me)
    return losses, reached


if __name__ == '__main__':
    reached = set()
    # Counterexample-guided symbolic synthesis: add positions reached by the
    # concrete float32 policy, then recompute the finite rule cover. No numeric
    # optimizer, gradient updates, random initialization, or self-play training.
    for iteration in range(30):
        path = build(reached)
        losses, seen = check(path)
        print(f'Exhaustive losses: {losses}', flush=True)
        if not losses:
            break
        reached.update(seen)
    else:
        raise RuntimeError('Synthesis did not converge')
