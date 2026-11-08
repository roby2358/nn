"""Exhaustive check: net (greedy, via the project's own PolicyNet.choose) vs every possible opponent line."""
import sys, collections
from tictactoe.ai.policy import PolicyNet
from tictactoe.board import Board
for path in sys.argv[1:]:
    net = PolicyNet.load(path)
    print(path)
    for me in 'XO':
        res = collections.Counter(); losses = []
        def dfs(b, hist):
            if b.game_over:
                res[b.winner or 'draw'] += 1
                if b.winner and b.winner != me: losses.append(hist)
                return
            moves = [net.choose(b, greedy=True)] if b.turn == me else [i for i in range(9) if b.cells[i] is None]
            for m in moves:
                n = Board(); n.cells = b.cells[:]; n.turn = b.turn
                n.play(m); dfs(n, hist + [m])
        dfs(Board(), [])
        print(f"  net as {me}: {dict(res)}   example loss: {losses[0] if losses else None}")
