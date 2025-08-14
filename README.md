# Tiny Go CNN
Bare minimum neural net to play plausible game of  Go

# Setup & Results
    Net:       7-layer CNN + fully connected output layer
    Games:     5000 pro games in SGF format
    Samples:   1 038 110 training positions
    Input:     2 channels (black & white stones, 19x19 each)
    Output:    score for every possible move on the 19×19 board
    Epoch:     10
    Loss:      2.1354
    CPU:       Intel© Core\u2122 i5-10400 CPU @ 2.90GHz × 6
    Time:     ~4hr
    Strength: ~17kyu
    
    Net:       11-layer CNN + fully connected output layer
    Games:     20000 pro games in SGF format
    Samples:   4 260 624 training positions
    Input:     2 channels (black & white stones, 19x19 each)
    Output:    score for every possible move on the 19×19 board
    Epoch:     6
    Loss:      2.81
    CPU:       Intel© Core\u2122 i5-10400 CPU @ 2.90GHz × 6
    Time:     ~60hr, (~2.5 days)
    Strength: ~10kyu (~5 kyu to ~14kyu, very imbalanced)
