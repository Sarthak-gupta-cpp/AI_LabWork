% AI Laboratory: Logical Planning - Prolog Plan Verifier
% File: planner.pl

% Facts describing the connectivity graph between warehouse locations
connected(a, b).
connected(b, a).
connected(b, c).
connected(c, b).

% Rule: The robot can move from X to Y if X and Y are directly connected
can_move(X, Y) :-
    connected(X, Y).

% Rule: A proposed movement is logically valid if supported by connectivity facts
valid_move(X, Y) :-
    connected(X, Y).

% Example queries:
% ?- can_move(a, b).    % Expected: true
% ?- can_move(a, c).    % Expected: false
% ?- valid_move(a, b).  % Expected: true
% ?- valid_move(b, c).  % Expected: true
% ?- valid_move(a, c).  % Expected: false

% Task 8: Rule-Based Logical Reasoning Chain
% Facts:
wet_road.

% Rules:
slippery :-
    wet_road.

reduce_speed :-
    slippery.

% Query:
% ?- reduce_speed.      % Expected: true
