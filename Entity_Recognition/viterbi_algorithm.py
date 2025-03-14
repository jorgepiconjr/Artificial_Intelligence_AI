from tabulate import tabulate

class prob():
    """Counts occurances for a pair of keys (either state to state or state to observation)
    as well as frequency of first state overall. Output are probability to transition."""

    def __init__(self):
        self.c={}
        self.n={}
        self.p={}

    def inc(self,k1,k2):
        self.c[(k1,k2)] = self.c.get((k1,k2),0) + 1
        self.n[k1] = self.n.get(k1,0) + 1

    def prob(self):
        for (k1,k2) in self.c:
            self.p[(k1,k2)] = float(self.c[(k1,k2)]) / float(self.n[k1])
        return self.p

def max_pos(l):
    """Returns the maximum and its position."""
    m = max(l)
    return(l.index(m),m)

def viterbi_alg(str, transition, emission):
    viterbi={} # dynamic programming matrix
    pos={}     # matrix to record path for backtracking
    obs = str.split()

    obs = [o.replace(".", "") for o in obs]
    states = list(set([s for (s,o) in emission]))
    # Init

    for s in states:
        viterbi[(s,0)]= float(s=="S")
    # Fill matrix
    for i in range(1,len(obs)):
        for j in states:
            # Fji = max F(r,i-1)*A(r,j)*B(j,i)
            (pos[(j,i)],viterbi[(j,i)]) = max_pos([viterbi[(r,i-1)]*transition.get((r,j),0.0)*emission.get((j,obs[i]), 0.0) for r in states])
    # Output table
    table = [[""]+ obs]
    for s in states:
        row = [s]
        for i in range(len(obs)):
            row.append(viterbi[(s,i)])
        table.append(row)
    s = "E"
    seq= ["E"]
    for i in range(len(obs)-1,0,-1):
        s = states[pos[(s,i)]]
        seq.insert(0,s)
    table.append([""]+seq)
    print()
    print(tabulate(table, headers="firstrow"))

def readTraining(fn):
    """Data format: First word presents state sequence. Remaining words are observations.
    For each character in the first word, there must be a word following.
    Start state is S, End state is E
    """
    t=prob()
    e=prob()
    print()
    print("HMM is trained on the following data:")
    for line in open(fn):
        print(line.strip())
        l = line.split()
        seq=list(l[0])
        obs=l[1:]
        if len(seq)!=len(obs):
            print("Format error in line %s"%(line))
        else:
            # Count transitions
            for i in range(len(seq)-1):
                t.inc(seq[i],seq[i+1])
            # Count emissions
            for i in range(len(seq)):
                e.inc(seq[i],obs[i])
    return (t.prob(), e.prob())

'''
Training HMM-Model, we want to calculate transition and emission probabilities
'''
transition={}
emission={}
(transition, emission) = readTraining("hmm_denver.txt")
observations = set([o for (s,o) in emission])

print()
print("Transitions:")
t = sorted(transition.items(), key=lambda x: x[1], reverse=True)
for ((s1,s2),v) in t:
    print("%s -> %s: %0.2f"%(s1,s2,v))

print()
print("Emissions:")
e = sorted(emission.items(), key=lambda x: x[1], reverse=True)
for ((s,o),v) in e:
    print("%s -> %s: %0.2f"%(s,o,v))

'''
Example
'''
sentence = "Oscar went to Denzel"
viterbi_alg("$ "+sentence+" $", transition, emission)