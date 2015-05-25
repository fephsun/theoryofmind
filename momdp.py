from __future__ import print_function

import math
import random
import os
import sys
import itertools
import cPickle

verbose = False

def setVerbose(newVal):
    """
    Sorta hacky, but I want to let a user of this package turn print statements
    on and off.
    """
    global verbose
    verbose = newVal

def logAdd(a, b):
    """
    Approximates log(exp(a) + exp(b))
    Handles cases when exp(a) or exp(b) overflows
    """
    if b > a:
        # a is always larger.
        a, b = b, a
    if a - b > 100:
        # Just a to a rounding error.
        return a
    return b + math.log(math.exp(a - b) + 1)

class MOMDP(object):
    """
    A class that works with APPL to do PO-/MO- MDP inference.  Uses the mako
    template library.

    The code outside of this class implements a grid world using MOMDP.
    TODO: Implement shuffling thing.
    """
    def __init__(self, trans, obs, reward, initial, nF, nP, nA, nO,
        discount=0.9, tau=100.0, solver='APPL'):
        """
        Let there be F fully-observed states (indexed 0 to F-1),
        P partially-observed states,
        O different obs, and A different actions.

        Then, trans(f, p)[a] is a table of [(newf, newp)] | prob.
        obs(f, p) is a table of obs | prob
        reward(f, p, a) = reward
        initial(f, p) = prob of initial p, given f
        """

        self.trans = trans
        self.obs = obs
        self.reward = reward
        self.initial = initial
        self.discount = discount
        self.tau = tau
        self.solver = solver

        # Populate max values.
        self.A = nA
        self.F = nF
        self.P = nP
        self.O = nO

    def valueIterate(self, tolerance=0.000001):
        """
        Runs value iteration on self.  Assumes this is an MDP - we will
        ignore the partially-observable stuff completely.
        """
        values = [0] * self.F
        stop = False
        iters = 0
        while (not stop):
            stop = True
            for state in xrange(self.F):
                bestAction = None
                bestValue = -1000000
                actions = self.trans(state, 0)
                for action in actions:
                    value = self.reward(state, 0, action)
                    for newState, prob in actions[action].iteritems():
                        value += prob * values[newState[0]] * self.discount
                    if value > bestValue:
                        bestValue = value
                        bestAction = action
                if abs(values[state] - bestValue) > tolerance:
                    stop = False
                values[state] = bestValue
            iters += 1

        # Convert `values` into self.vectors
        self.vectors = {}
        for idx, value in enumerate(values):
            self.vectors[idx] = [[value]]

    def pathProb(self, path):
        """
        Calculates the probability that an agent will follow path.
        path is a list (location, aux) of states visited.
        Calculates P(location | aux).

        Assumes that loadPolicy has already been called successfully. 

        Performs a random estimate - call many times and average to get an
        accurate result.  There are potentially two sources of randomness: 
        which action the agent chooses and what observation the agent gets.
        If every action results in a different observable state, the first
        source is eliminated.  If every observation is fully deterministic,
        the second source is eliminated.

        Maybe I'll implement a better pathProb that gets the right answer
        on the first try, but that's for later.
        """
        # Establish initial state distribution.
        estState = []
        for s in range(self.P):
            estState.append(self.initial(path[0][0], s))
        logProb = 0
        for step in range(1, len(path)):
            # Calculate a softmax probability that the agent uses each alpha
            # vector, then sort by action.
            lastF = path[step-1][0]
            lastP = path[step-1][1]
            thisF = path[step][0]
            thisP = path[step][1]

            # These are log probs.
            actionProbs = [0.0]*self.A
            totalWeight = float('-inf')
            maxScore = float('-inf')
            for action in range(self.A):
                score = self.valueLookAhead(lastF, estState, action)
                maxScore = max(score, maxScore)
                actionProbs[action] = self.tau * score
                totalWeight = logAdd(totalWeight, self.tau * score)
            # Tally up the probability that the agent goes to the correct state.
            pTrans = 0
            actionTable = {}
            for action in range(self.A):
                nextSTable = self.trans(lastF, lastP)[action]
                if not (thisF, thisP) in nextSTable:
                    continue
                pThisAction = nextSTable[(thisF, thisP)] * \
                    math.exp(actionProbs[action] - totalWeight)
                actionTable[action] = pThisAction
                pTrans += pThisAction
            if pTrans == 0:
                return float('-inf')
            logProb += math.log(pTrans)

            # Choose which action we are taking.
            for action in actionTable:
                actionTable[action] /= pTrans
            thisAction = randomSample(actionTable)  #random!

            # Update the agent's guess of the hidden states.
            nextEstState = [0.0]*self.P
            thisObs = randomSample(self.obs(lastF, lastP))    #random!
            for guessP in range(self.P):
                # What is the probability we are in state guessP?
                pGuessP = estState[guessP] * self.obs(lastF, guessP)[thisObs]
                # Given that we are in state guessP, what is the probability that
                # we move to each new state in P?
                newStates = self.trans(lastF, guessP)[thisAction]
                for newState, prob in newStates.iteritems():
                    if newState[0] == thisF:
                        nextEstState[newState[1]] += pGuessP * prob
            # Normalize nextEstState.
            estState = [i/sum(nextEstState) for i in nextEstState]
        return logProb

    def valueLookAhead(self, state, estState, action):
        """
        A helper function for path probability estimation.
        Finds the value of taking action with observable state state
        and partially-observable state distribution estState.

        Relies on the existance of self.vectors.
        """
        # Get the state distribution, assuming we take action.
        newDist = {}
        for pState in range(self.P):
            for newState, prob in self.trans(state, pState)[action].iteritems():
                newF, newP = newState
                if newState not in newDist:
                    newDist[newF] = [0.0] * self.P
                # Note that newDist[newF] is a (not-normalized)
                # state probability distribution.
                newDist[newF][newP] += prob * estState[pState]

        # For each possible newF, calculate the maximum value.
        maxValue = -float('inf')
        for newF, dist in newDist.iteritems():
            normDist = [x/sum(dist) for x in dist]
            for vector in self.vectors[newF]:
                dotProduct = sum(vector[i] * normDist[i] for i in range(self.P))
                if dotProduct > maxValue:
                    maxValue = dotProduct

        rewardValue = 0
        for pState in range(self.P):
            rewardValue += self.reward(state, pState, action) * estState[pState]
        return maxValue + rewardValue

    def mostLikelyPath(self, start, length=100):
        """
        Returns the most likely path of this POMDP.  Uses pathProb.
        start is a (f, p) starting state pair.
        """
        path = [start]
        for step in range(length):
            bestValue = -100000000
            bestAction = None
            for action in range(self.A):
                pathValue = self.valueLookAhead(path[-1][0], [1], action)
                # print((self.stateToCoord[nextState[0]], pathValue))
                if pathValue > bestValue:
                    bestValue = pathValue
                    bestAction = action
            possibleNextStates = self.trans(path[-1][0], path[-1][1])[bestAction]
            path.append(randomSample(possibleNextStates))
        return path

    def simplexSample(self, paths, holeFiller, dimension, density=101, part=None):
        """
        We have a MOMDP in which several reward values are unknown.  We
        observe an agent following this MOMDP travel on several paths, where
        each path is a list of (fullyObs, partiallyObs) states.  This function
        calculates the probability of each reward value combination, given the
        paths.

        There are `dimension` unknown reward values, sampled at `density` samples
        between 0 and 1.  holeFiller (which you provide) handles filling the
        unknown reward values.

        holeFiller takes in a list, of length dimension, of floats [0, 1],
        and returns a dictionary of {(state, action): value}


        part tells simplexSample to only do a portion of the sampling task.  For
        example, if part = [0, 1, 2, 3, 4], simplexSample will only return the
        first 5 samples.  If part = None, we will do the entire task.
        """
        oneDRange = [1.0*x/(density-1) for x in range(density)]
        allValues = list(itertools.product(oneDRange, repeat=dimension))
        if part is None:
            inValues = allValues
        else:
            inValues = []
            for idx in part:
                inValues.append(allValues[idx])

        logProbs = {}
        for inValue in inValues:
            if verbose:
                print(inValue)
            # Fill in missing reward values
            self.reward = holeFiller(inValue)

            if self.solver == 'APPL':
                # Do inference and stuff
                self.toPomdpX()
                self.loadPolicy()
            elif self.solver == 'mdp':
                self.valueIterate()
            else:
                assert False, "Invalid solver."
            thisLogProb = 0
            for path in paths:
                thisLogProb += self.pathProb(path)
            logProbs[inValue] = thisLogProb
        return logProbs

def randomSample(probs):
    """
    Helper function:
    Given a dictionary probs[outcome] = probability.
    Returns a random outcome.
    """
    rnum = random.uniform(0, 1)
    p_so_far = 0
    for outcome, prob in probs.iteritems():
        if p_so_far + prob >= rnum:
            return outcome
        else:
            p_so_far += prob
    assert False, "Unexpected case in random_sample"

def myCombinations(iterable, r):
    """
    The default itertools combinations doesn't sort tuples correctly.
    So, here's my own.
    """
    for perm in itertools.permutations(iterable, r):
        if sorted(perm) == list(perm):
            yield perm


class GridMOMDP(MOMDP):
    """
    A subclass of MOMDP that parses grid MOMDPs from strings.
    """

    def __init__(self, grid, numRewards, discount, tau, moveCost, oneAtATime=False):
        """
        grid is a list of strings representing the world.  For example:
        ["1 x 2",
         "2 x 1",
         "11S22"]
        We recognize 1, 2, S, and x right now.

        Creates the following attributes:
        self.coordToState[(x, y, rewardsTaken)] -> state number
        self.stateToCoord[state number] -> (x, y, rewardsTaken)
        self.holes1 - list of reward1 locations
        self.holes2 - list of reward2 locations
        self.exit - (x, y) of exit location
        self.rTaken - a list of all possible combinations of rewards the agent
            may take.
        self.oneAtATime - Whether the agent must visit the exit in between
            picking up rewards.
        """
        self.holes1 = []
        self.holes2 = []
        self.oneAtATime = oneAtATime
        self.numRewards = numRewards
        self.moveCost = moveCost
        # An auxillary variable that points to the reward the agent is currently
        # holding, if the agent is only allowed to pick up one reward at a time.
        if self.oneAtATime:
            self.holdingPossibilities = [-1] + range(numRewards)
        else:
            self.holdingPossibilities = [-1]
        walls = []
        self.exit = None
        height = len(grid)
        width = len(grid[0])
        for y in range(height):
            for x in range(width):
                if grid[height-1-y][x] == '1':
                    self.holes1.append((x, y))
                if grid[height-1-y][x] == '2':
                    self.holes2.append((x, y))
                if grid[height-1-y][x] == 'x':
                    walls.append((x, y))
                if grid[height-1-y][x] == 'S':
                    self.exit = (x, y)

        allHoles = self.holes1 + self.holes2
        thisTrans, self.coordToState, self.stateToCoord, self.rTaken = \
            self.makeGrid(width, height, allHoles, self.exit, numRewards, 1, walls)
        # Give the dead-end state a coordinate, equal to the exit.
        self.stateToCoord[len(self.stateToCoord)] = (self.exit[0], self.exit[1], (), -1)
        thisObs = lambda a, b: {0: 1}
        thisReward = lambda a, b, c: 0
        thisInitial = lambda a, b: 1

        super(GridMOMDP, self).__init__(thisTrans, thisObs, thisReward, thisInitial, 
            len(self.stateToCoord), 1, 5, 1, 
            discount, tau, solver='mdp')


    def makeGrid(self, width, height, rewardLocs, exit, nPick=1, nAux=1, walls=[]):
        """
        Makes a grid MOMDP.  Returns a transition function, with five
        possible actions (0-3; left, up, down, right; 4 = pick up reward)
        Walls is a list of coordinates that are not occupiable.

        rewardLocs is a list of places that can have rewards.
        exit is the place that the agent must visit at the end of the game.
        nPick is the number of rewards the agent is allowed to pick up.

        nAux is the number of auxillary, partially-observed states available.
        You need to specify for yourself what these aux states do, after you
        call this function.

        Returns a transition dictionary, with a bunch of other stuff.

        Warning: number of states is O(width height nAux len(rewardLocs)**nPick)
        Many possible rewards, with the option to pick up rewards, will result
        in HUGE MOMDPs.
        """
        # Make mapping from coordinate (x, y, (takenreward1, takenreward2, ...))
        # to state number, and vice-versa.
        rTaken = iter([(),])
        for nPicked in range(1, nPick+1):
            rTaken = itertools.chain(rTaken, 
                myCombinations(rewardLocs, r=nPicked)
            )
        # Iterators are hard to reset, so we list it.
        rTaken = list(rTaken)

        # Mappings from state to coordinates, vice-versa
        coordToState = {}
        stateToCoord = {}
        stateIdx = 0
        for x in range(width):
            for y in range(height):
                for stuff in rTaken:
                    for holding in self.holdingPossibilities:
                        coordToState[(x, y, stuff, holding)] = stateIdx
                        stateToCoord[stateIdx] = (x, y, stuff, holding)
                        stateIdx += 1
        self.deadEndState = stateIdx

        # Actually make the transition function
        def trans(f, p): 
            aux = p
            (x, y, stuff, holding) = stateToCoord[f]
            actionMap = {}
            default = {(f, aux): 1}
            # Make the transition dictionary if the dead-end state (state width*height)
            if f == self.F-1:
                for action in range(5):
                    actionMap[action] = default
                return actionMap

            # Otherwise, determine directions of motion, etc.   
            for i in range(4):
                actionMap[i] = default
            if x != 0 and ((x-1, y) not in walls):
                actionMap[0] = {(coordToState[(x-1,y,stuff, holding)], aux): 1}
            if x < width-1 and ((x+1, y) not in walls):
                actionMap[1] = {(coordToState[(x+1,y,stuff, holding)], aux): 1}
            if y != 0 and ((x, y-1) not in walls):
                actionMap[2] = {(coordToState[(x,y-1,stuff, holding)], aux): 1}
            if y < height-1 and ((x, y+1) not in walls):
                actionMap[3] = {(coordToState[(x,y+1,stuff, holding)], aux): 1}
            # What happens when the agent uses action 4?
            if (x, y) == exit:
                # Some cases, depending on self.oneAtATime
                if not self.oneAtATime:
                    # The agent is leaving.
                    actionMap[4] = {(self.deadEndState, aux): 1}
                else:
                    # The agent is dropping off a reward.  holeFiller will
                    # take care of the reward value.
                    if len(stuff) >= nPick:
                        # The agent is not allowed to pick up more stuff
                        actionMap[4] = {(self.deadEndState, aux): 1}
                    else:
                        # The agent drops off the object.
                        actionMap[4] = {(coordToState[(x,y,stuff, -1)], aux): 1}
            elif (x, y) not in rewardLocs:
                # No reward to pick up.  Do nothing.
                actionMap[4] = default
            elif (x, y) in stuff:
                # This reward has already been used.  Do nothing.
                actionMap[4] = default
            elif len(stuff) >= nPick or (holding != -1 and holding < len(stuff)
                and self.oneAtATime):
                # The agent has its hands full.
                actionMap[4] = default
            else:
                # The agent is allowed to pick up an object.
                newStuff = tuple(sorted(list(stuff) + [(x, y)]))
                if self.oneAtATime:
                    newHoldingIdx = newStuff.index((x, y))
                else:
                    newHoldingIdx = -1
                actionMap[4] = {(coordToState[(x, y, newStuff, newHoldingIdx)], aux): 1}
            return actionMap

        # Man, I'm outputting a lot of stuff.
        # coordToState[(x, y, rewardsLeft, holding)] -> index of this state
        # stateToCoord[index] -> (x, y, rewardsLeft, holding)
        # rTaken is a list of all possible combinations of leftover rewards.
        return (trans, coordToState, stateToCoord, rTaken)

    def holeFiller(self, inList):
        """
        Makes a rewards mapper (fullState, partialState, action) -> reward, given
        some sampling parameters.
        """
        def rewardFn(f, p, a):
            x, y, rewardsTaken, holding = self.stateToCoord[f]
            if f == self.deadEndState:
                return 0
            if (x, y) != self.exit or a != 4:
                return -self.moveCost
            # Reminder: rCombo is like ((1, 2), (5, 3))
            totalValue = 0
            for rewardLoc in rewardsTaken:
                if rewardLoc in self.holes1:
                    totalValue += inList[0]
                elif rewardLoc in self.holes2:
                    totalValue += 1 - inList[0]
                else:
                    assert False, "Location does not correspond to any reward."
            return totalValue - self.moveCost
        return rewardFn

    def oneAtATimeHoleFiller(self, inList):
        """
        Makes rewards mapper, for one-at-a-time sampling.
        """
        def rewardFn(f, p, a):
            x, y, rCombo, holding = self.stateToCoord[f]
            if f == self.deadEndState:
                return 0
            if (x, y) != self.exit or a != 4:
                return -self.moveCost
            if holding >= len(rCombo) or holding == -1:
                # Invalid picked-up thing, or not holding anything.
                # No prize for you.
                return -0.1
            if rCombo[holding] in self.holes1:
                value = inList[0]
            elif rCombo[holding] in self.holes2:
                value = 1 - inList[0]
            else:
                assert False, "Location does not correspond to any reward."
            return value - self.moveCost
        return rewardFn

    def encodePath(self, path):
        """
        Converts a human-readable path into a list of states visited.
        """
        codedPath = []
        for x, y, pickedRewards, holding in path:
            rewardsList = sorted(list(pickedRewards))
            codedPath.append((self.coordToState[(x, y, tuple(rewardsList), holding)], 0))
        return codedPath

    def testGrid(self, path, density, part):
        """
        path is a list of coordinates that the agent visits.  numRewards is
        the number of rewards the agent is allowed to pick up.

        Returns a dictionary of posterior samples.  {Value of red reward -> log P(value)}
        """
        codedPath = self.encodePath(path)
        if self.oneAtATime:
            curriedHoleFiller = lambda inList: self.oneAtATimeHoleFiller(inList)
        else:
            curriedHoleFiller = lambda inList: self.holeFiller(inList)
        results = self.simplexSample([codedPath], curriedHoleFiller, 1, density, part)
        return results
    # simplexPlot(results.items())

def simplexPlot(results):
    """
    Plot and log the results of a 1D simplexSample operation.
    `results` is the output of simplexSample.
    """
    _, hole1Values, logProbs = getStatistics(results.items())
    import matplotlib.pyplot as plt
    plt.plot(hole1Values, logProbs)
    plt.show()

def getStatistics(results):
    """
    Input is list of [ [[value], prob], ... ]

    Returns a variety of statistics on the posterior distribution.
    """
    results.sort(key=lambda x: x[0][0])
    hole1Values = [x[0][0] for x in results]
    logProbs = [x[1] for x in results]
    maxLogProb = max(logProbs)
    probs = [math.exp(x - maxLogProb) for x in logProbs]
    probs = [i / sum(probs) for i in probs]
    print(probs)
    expectedValue = sum(probs[i]*hole1Values[i] for i in xrange(len(probs)))
    stats = {}
    stats['avg'] = expectedValue
    stats['sd'] = sum(
        probs[i] * (expectedValue - hole1Values[i])**2 for i in range(len(probs))
    ) ** 0.5
    stats['map'] = hole1Values[max(xrange(len(probs)), key = lambda i: probs[i])]
    stats['pavg'] = probs[int(stats['avg']*(len(results) - 1))]
    stats['pmap'] = probs[int(stats['map']*(len(results) - 1))]
    return stats, hole1Values, logProbs

def localRunner(params, nSamples=101, part=range(101)):
    """
    A shortcut function to run an experiment.
    """
    grid = params['grid']
    path = params['path']
    nPickUp = params['nPickUp']
    oneAtATime = params['oneAtATime']
    discount = params['discount']
    tau = params['tau']
    moveCost = params['moveCost']

    momdp = GridMOMDP(grid, nPickUp, discount, tau, moveCost, oneAtATime)
    results = momdp.testGrid(path, nSamples, part)
    return results
