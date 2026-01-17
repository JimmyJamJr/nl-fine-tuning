Is 160M reasonable or should we choose another model size?

How should I decide how much to mix in the pretraining set?

What is a good interval to move on to the next increment in the curriculumn?

Should our curriculumn be mixed lookahead size or a max lookahead size?

How do we decide the max lookahead size to do in the pre pretraining?


If Pretraining is oriented around learning langauge as a whole, does it make sense to do this task in pre pretraining. 

MODEL
Pythia 160M

PLAN
1. NL Curriculm on Search Task 0.1-1.0 at 0.98 before moving (lookahead = 128)
2. Train on C4 using configuration from paper (Between Circuits and Chomsky)
3. Compare Performance


POSSIBLE VARIATIONS
Curriculum - Pretraining (0% mixed in)
Curriculum - Pretraining (X% mixed in) (like 1% 5% etc.)
Pretraining With (X% mixed in)


# Meeting Notes
- Catastrophic forgetting so we need to do the max lookahead thing so that we do not need to worry about catastrophic forgetting

- Changing alpha so that it is proportional to lookahead

- Make sure to plot the loss function in log space


teach it to do search so it should reason first then it should 

- Dictionary with vocab size is important with curriculumn, then 

Does it learn natural language better
Does it learn this search better 



NEXT TOKEN PREDICTION BUT THEN UPWEIDGHT THE ANSWER TOKEN
COMPUTE OPTIMAL AND NUMBER OF DATA OPTIMAL FOR A CERTAIN MODEL SIZE

70B 1.4Trillion 

20 tokens per parameter

1.7 Tokens per parameter

Increment percentage as we go on