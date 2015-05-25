import momdp
import cPickle
momdp.setVerbose(True)

stimuli = cPickle.load(open('stimuli.pickle', 'rb'))
# Take some arbitrary stimulus from the list of stimuli, and run it. 
gridName, grid, agentPreference, nPick, oneAtATime, path, readablePath = stimuli[12]
print 'World name: ' + gridName
print 'Agent preference for 1: ' + str(agentPreference)
print 'Agent can pick up ' + str(nPick) + ' balls'
if oneAtATime:
    print 'Agent can only hold one ball at a time.'
else:
    print 'Agent can hold multiple balls.'
print 'World map:'
for line in range(len(grid)):
    print grid[line]

discount = 0.999
beta = 100
moveCost = 0.03
nSamples = 11
mymomdp = momdp.GridMOMDP(grid, nPick, discount, beta, moveCost)
out = mymomdp.testGrid(readablePath, nSamples, range(nSamples))
momdp.simplexPlot(out)