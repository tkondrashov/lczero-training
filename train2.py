# selfplay with two networks trying to achieve certain values for properties, eg:
# - p1 wants winner=1, p2 wants winner=1 (both sides want white to win)
# - p1 wants winner=1, p2 wants winner=0 (black wants to draw)
# - p1 wants winner=1, p2 wants winner=-1 (both side want to win)
# - p1 wants bishop_pair_us=1, p2 wants bishop_pair_us=0

# i_win 1 winner 1 up(piece_type) 1 count(piece_type) 1 color 1
#       0        0                0                   0       0
#      -1       -1               -1                          -1

goals = [
    # W win: W&!D
    # L lose: L&!D
    # D draw: !W&!L&D
    # C complicated: !W&!L&!D
    # S simple: W|L|D
    # R random!
#   p1 p2
    W  W
    W  L
    W  D
    W  C
    W  S
    W  R
    L  L
    L  D
    L  C
    L  S
    L  R
    D  D
    D  C
    D  S
    D  R
    C  C
    C  S
    C  R
    S  S
    S  R
    R  R
]

# simplify(threshold) stands for a function that tries to get the simplest model with accuracy > threshold

model = Random()
while properties:
    complex_data = model.selfplay(goals)
    while data:
        intuition = train((win, draw), data).simplify(.9)
        complex_data += model.predict(data)
        if model.accuracy != 1:
            simple_data = data.filter(train_Y == test_Y)
            logic = train((win, draw), simple_data).simplify(1)
            data += simple_model.predict(data)
            data = data.filter(train_Y != test_Y)
            properties.append(property)
