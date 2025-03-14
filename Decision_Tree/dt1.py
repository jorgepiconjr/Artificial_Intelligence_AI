import numpy as np

def openfile(path, fname):
    """opens the file at path+fname and returns a list of examples and attribute values.
    examples are returned as a list with one entry per example. Each entry then
    is a list of attribute values, one of them being the class label. The returned list attr
    contains one entry per attribute. Each entry is a list of possible values or an empty list
    for numeric attributes.
    """
    datafile = open(path + fname, "r")
    examples = []
    for line in datafile:
        line = line.strip()
        line = line.strip('.')
        # ignore empty lines. comments are marked with a |
        if len(line) == 0 or line[0] == '|':
            continue
        ex = [x.strip() for x in line.split(",")]
        examples.append(ex)
    attr = []
    for i in range(len(examples[0])):
        values = list({x[i] for x in examples})  # set of all different attribute values
        if values[0].isdigit():  # if the first value is a digit, assume all are numeric
            attr.append([])
        else:
            attr.append(values)

    return examples, attr


def calc_entropy(examples, cls_index):
    """calculates the entropy over all examples. The index of the class label in the example
    is given by cls_index. Can also be the index to an attribute.
    """
    global attr
    # get attributes of examples with index cls_index
    example_classifications = [example[cls_index] for example in examples]
    # get unique counts of example_attributes
    unique, counts = np.unique(example_classifications, return_counts=True)

    # normalize counts to total number of examples for getting probs
    probs = counts/len(examples)
    # print(probs)

    entropy = -np.sum(probs*np.log2(probs))

    return entropy


def calc_ig(examples, attr_index, cls_index):
    """Calculates the information gain over all examples for a specific attribute. The
    class index must be specified.

    uses calc_entropy
    """
    global attr
    # get attributes of examples with index attr_indx
    example_attributes = [example[attr_index] for example in examples]

    # get unique counts of example_attributes
    unique, counts = np.unique(example_attributes, return_counts=True)

    # example split
    remainder = 0
    for j in range(len(unique)):
        # get all examples with attribute unique[j]
        examples_unique = [example for example in examples if example[attr_index] == unique[j]]
        remainder += len(examples_unique)/len(examples)*calc_entropy(examples_unique, cls_index)
    return calc_entropy(examples, cls_index) - remainder


def majority(examples, attr_index):
    """Returns the value of attribute "attr_index" that occurs the most often in the examples."""
    # create a flat list of all attribute values (with duplicates, so we can count)
    attr_vals = [example[attr_index] for example in examples]

    # among all unique attribute values, find the maximum with regards to occurrence in the attr_vals list

    maj_attr_val = max(set(attr_vals), key=attr_vals.count)
    return maj_attr_val

def choose_best_attr(examples, attr_avail, cls_index):
    """Iterates over all available attributes, calculates their information gain and returns the one
    that achieves the highest. attr_avail is a list of booleans corresponding to the list of attributes.
    it is true if the attribute has not been used in the tree yet (and is not numeric).
    """
    igs = []  # list of information gains for each attribute

    for j in range(len(attr)):
        if attr_avail[j]:
            igs.append(calc_ig(examples, j, cls_index))
        else:
            igs.append(-1)

    return igs.index(max(igs))  # return index of the attribute with highest IG


def dtree_learning(examples, attr_avail, default, cls_index):
    """Implementation of the decition tree learning algorithm. Receives the remaining examples,
    the remaining attributes (as boolean list), the default label and the index of the class label in
    the attribute vector. Returns the root node of the decision tree.

    Each tree node is a tuple where the first entry is the index of the
    attribute that has been used for the split. It is "None" for leaf nodes.
    The second entry is a list of subtrees of the same format. The subtrees are ordered in the
    same way as the attribute values in "attr". For leaf nodes, the second entry is the predicted class.

    uses choose_best_attr, majority, dtree_learning
    """
    global attr
    if len(examples) == 0:
        # is leaf
        # no examples left -> majority class prediction of parent
        return [None, default]
    elif len({x[cls_index] for x in examples}) == 1:
        # is leaf
        # uniform classes
        return [None, examples[0][cls_index]]
    elif attr_avail.count(True) == 0:
        # is leaf
        # no attribute left to split -> majority class prediction
        return [None, majority(examples, cls_index)]
    else:
        best = choose_best_attr(examples, attr_avail, cls_index)
        tree = [best, []]
        for v in attr[best]:
            examples_v = [x for x in examples if x[best] == v]
            # why is it important to make a copy and not change it directly?
            new_attr_avail = attr_avail.copy()
            new_attr_avail[best] = False
            subtree = dtree_learning(examples_v, new_attr_avail, majority(examples, cls_index), cls_index)
            tree[1].append(subtree)
        return tree


def dtree_classify(dtree, x):
    """Classifies a single example x using the given decision tree. Returns the predicted class label.
    """
    # attribute index of splitting attribute
    attr_split_index = dtree[0]

    if attr_split_index is not None:
        # get attribute value for example
        example_split_attr = x[attr_split_index]

        # subtree position
        subtree_pos = attr[attr_split_index].index(example_split_attr)

        return dtree_classify(dtree[1][subtree_pos], x)  # descend into subtree recursively
    else:
        return dtree[1]


def dtree_test(dtree, examples, cls_index):
    """Classify all examples using the given decision tree. Prints the achieved accuracy."""
    correct = 0
    for j in range(len(examples)):
        if dtree_classify(dtree, examples[j]) == examples[j][cls_index]:
            correct += 1

    print("{} out of {} correct ({:.2f}%)".format(correct, len(examples), correct / len(examples) * 100))


path = "./"  # directory of your data
datafile = "adult.data.txt"
testfile = "adult.test.txt"
examples, attr = openfile(path, datafile)  # load the training set
test, test_attr = openfile(path, testfile)  # load the test set
cls_index = len(attr) - 1  # the last attribute is assumed to be the class label
# attr_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "class"]

attr_avail = []  # marks which attributes are available for splitting (not numeric and not the class label)
for i in range(len(attr)):
    # the list attr[i] contains all possible values of attribute i. It is empty for numeric attributes.
    attr_avail.append(len(attr[i]) > 0 and i != cls_index)

# print(attr_avail)
dtree = dtree_learning(examples, attr_avail, [], cls_index)

print("Before removal of unknowns: ")
print("Training Set")
dtree_test(dtree, examples, cls_index)
print("Test Set")
dtree_test(dtree, test, cls_index)

# Extra task removal of unknowns
examples_removed = []
test_removed = []

for example in examples:
    # print(example.__contains__("?"))
    if not example.__contains__("?"):
        examples_removed.append(example)

for example in test:
    if not example.__contains__("?"):
        test_removed.append(example)

# print(len(examples_removed))
# print(len(test_removed))
dtree = dtree_learning(examples_removed, attr_avail, [], cls_index)

print("After removal of unknowns: ")
print("Training Set")
dtree_test(dtree, examples_removed, cls_index)
print("Test Set")
dtree_test(dtree, test_removed, cls_index)