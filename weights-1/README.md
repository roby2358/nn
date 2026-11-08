# Deterministic tic-tac-toe weights

Run `python build_weights.py` to construct `weights.npz` for the existing
10-input, 64-hidden-neuron, 9-output `PolicyNet` in the neighboring
`tictactoe` project. Requires NumPy. No source changes to the game are needed.

The builder enumerates game positions, uses exact minimax values to identify
nonlosing moves, and compiles occupied-square conjunctions into tanh neurons.
Sixteen neurons encode immediate wins and blocks. Sixteen additional neurons
encode strategic rules; the other 32 neurons have zero output weights.
The output weights are calculated from the rules, without Adam, gradient
descent, numerical fitting, random sampling, or training games.

The builder checks every opponent continuation using the actual float32
`PolicyNet.choose` implementation. If there is a loss, its reachable positions
are added to the symbolic constraints and the rules are reconstructed.
Only a construction with zero exhaustive losses is accepted.

To check independently from this directory in PowerShell:

```powershell
$env:PYTHONPATH = '../tictactoe/src'
python ../tictactoe/verify.py weights.npz
```

Copy `weights.npz` into the game directory and launch the game from that
directory, or press Load if it is already running. The guarantee applies to
greedy play through `PolicyNet.choose` from an empty board, as either side,
against every legal opponent continuation. Sampling moves or modifying the
weights with Train is outside that verified policy.
