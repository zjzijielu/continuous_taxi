import numpy as np
import sys
from six import StringIO
import json
from gym import spaces, utils
from gym.envs.toy_text import discrete


MAP = [
    "+---------+",
    "|R:B: : : |",
    "| : : :H: |",
    "|_: :F:S: |",
    "| : : | : |",
    "|A: : |T: |",
    "+---------+",
]


class StochasticTaxiEnv2(discrete.DiscreteEnv):
    """
    Time = 12 hours = 48 quarters
    Go from one grid to another takes 1 quarter

     +---------+
    "|R:B: : : |
    "| : : :H: |
    "|_: :F:S: |
    "| : : | : |
    "|A: : |T: |
    "+---------+

    R:Residential Area
    A:Airport
    F:Financial Center
    S:Shopping center
    T:Tech Park
    H:Hotel
    B:Beginning point

    rendering:
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters: locations

    """

    metadata = {'render.modes': ['human', 'ansi']}

    ## loading request table
    request = []

    def __init__(self):
        self.desc = np.asarray(MAP,dtype='c')
        self.locs = locs = [(0,0), (1,3), (2,2), (2,3), (4,0), (4,3), None]

        with open('request_table_v2.json', 'r') as f:
            request = json.load(f)

        self.request = request

        nS = 60025
        nR = 5
        nC = 5
        maxR = nR - 1
        maxC = nC - 1

        nA = 9
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        for time in range(48):
            for row in range(5):
                for col in range(5):
                    for pidx in range(7):
                        for didx in range(7):

                            state = self.encode(time, row, col, pidx, didx)
                            for a in range(nA):
                                # defaults
                                newrow, newcol = row, col
                                passidx, destidx = pidx, didx
                                reward = -1
                                done = False
                                taxiloc = (row, col)
                                if a==0: # go down
                                    if self.desc[row+2, 2*col+1] != b"_":
                                        newrow = min(row+1, maxR)
                                    if newrow == row:
                                        reward = -10
                                    newtime = time + 1
                                elif a==1: # go up
                                    if self.desc[row, 2*col+1] != b"_":
                                        newrow = max(row-1, 0)
                                    if newrow == row:
                                        reward = -10
                                    newtime = time + 1
                                elif a==2 and self.desc[1+row,2*col+2]==b":": # go right
                                    newcol = min(col+1,maxC)
                                    newtime = time + 1
                                elif a==2 and self.desc[1 + row, 2*col+2] == b"|":
                                    reward = -10
                                    newtime = time + 1
                                elif a==3 and self.desc[1+row,2*col]==b":": # go left
                                    newcol = max(col-1, 0)
                                    newtime = time + 1
                                elif a==3 and self.desc[1+row,2*col]==b"|":
                                    reward = -10
                                    newtime = time + 1
                                elif a==4: # pickup the first request
                                    if destidx!=6 or self.request[time][str((row,col))]==None:
                                        reward = -10
                                    else:
                                        #print(self.request[time])
                                        destidx = self.request[time][str((row,col))][0]
                                        #print(time, row, col, destidx)
                                        #destidx = destidx[0]
                                        passidx = locs.index((row, col))
                                    newtime = time
                                elif a==5: # pickup the second request
                                    if destidx!=6 or self.request[time][str((row,col))]==None:
                                        reward = -10
                                    elif len(self.request[time][str((row,col))]) < 2: # only one request available:
                                        reward = -10
                                    else:
                                        destidx = self.request[time][str((row,col))][1]
                                        passidx = locs.index((row, col))
                                    newtime = time
                                elif a==6: # pickup the third request
                                    if destidx!=6 or self.request[time][str((row,col))]==None:
                                        reward = -10
                                    elif len(self.request[time][str((row,col))]) < 3: # only one request available:
                                        reward = -10
                                    else:
                                        destidx = self.request[time][str((row,col))][2]
                                        passidx = locs.index((row, col))
                                    newtime = time
                                elif a==7: # dropoff
                                    if destidx != 6 and (taxiloc == locs[destidx]) and passidx != 6:
                                        reward = 5   * sum([abs(locs[passidx][x] - locs[destidx][x]) for x in range(2)])
                                        destidx = 6 # no destination
                                        passidx = 6 # no passenger position index
                                    else:
                                        reward = -10
                                    newtime = time
                                elif a==8: # Stay still
                                    if destidx!=6:
                                        reward = -5
                                    newtime = time + 1
                                if newtime==48:
                                    done = True
                                    if taxiloc != locs[destidx] and destidx != 6 and passidx != 6:
                                        reward = 4 * sum([abs(locs[passidx][x] - locs[destidx][x]) for x in range(2)])
                                        cost = 2 * sum([abs(locs[destidx][x] - locs[1][x]) for x in range(2)])
                                        reward = reward - cost
                                newstate = self.encode(newtime, newrow, newcol, passidx, destidx)
                                #if time==4 and row==0 and col==0:
                                #    print(time, state)
                                P[state][a].append((1.0, newstate, reward, done))

                                # setting initial state as 7AM, at home, no pickup, no destination
        isd = np.zeros(nS)
        isd[self.encode(0, 0, 1, 6, 6)] = 1

        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)

    def encode(self, time, newrow, newcol, passidx, destidx):
        # (48) 5, 5, 7, 7
        i = time
        i *= 5
        i += newrow
        i *= 5
        i += newcol
        i *= 7
        i += passidx
        i *= 7
        i += destidx
        return i

    def decode(self, i):
        out = []
        out.append(i % 7)
        i = i // 7
        out.append(i % 7)
        i = i // 7
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        out.reverse()
        return out

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        # print(out)
        time, taxirow, taxicol, passidx, destidx = self.decode(self.s)

        # print(time, taxirow, taxicol, destidx)
        def ul(x):
            return "_" if x == " " else x
        if destidx ==6:  # no passenger in taxi
            # print(out[1 + taxirow][2 * taxicol + 1])
            out[1 + taxirow][2 * taxicol + 1] = utils.colorize(out[1 + taxirow][2 * taxicol + 1], 'yellow',
                                                               highlight=True)
        else:  # passenger in taxi
            # print(ul(out[1 + taxirow][2 * taxicol + 1]))
            # highlight destination
            out[1 + taxirow][2 * taxicol + 1] = utils.colorize(ul(out[1 + taxirow][2 * taxicol + 1]), 'green',
                                                                   highlight=True)
            di, dj = self.locs[destidx]
            out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], 'magenta')

            # highlight pickup location
            pi, pj = self.locs[passidx]
            out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'red')
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write(
                "  ({})\n".format(
                    ["South", "North", "East", "West", "Pickup 1st", "Pickup 2nd", "Pickup 3rd", "Dropoff", "Stay Still"][self.lastaction]))
        else:
            outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            return outfile