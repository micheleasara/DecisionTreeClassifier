import numpy as np
import matplotlib.pyplot as plt

att_name = {0: 'x-box', 1: 'y-box', 2: 'width', 3: 'high', 4: 'onpix', 5: 'x-bar', 6: 'y-bar', 7: ' x2bar',
                8: 'y2bar', 9: 'xybar', 10: 'x2ybr', 11: 'xy2br', 12: 'x-ege', 13: 'xegvy', 14: 'y-ege', 15: 'yegvx'}
class Dataset(object):
    """ Class to read, store, visualise data

    Methods
    -------
        load_first(path)
            Loads the data from given file
        get_labels(data)
            Extracts the labels portion from the dataset and stores as a class object.
        get_attributes(data)
            Extracts the attributes portion from the dataset and stores as a class object.
        set_up_count_dict(label array)
            This creates a dictionary of label category and frequency and stores as a class object.
        att_range(attribute index)
            Returns a tuple of the numerical range for a particular attribute index i.e. (min_val, max_val).
        range_graph()
            Generates a plot to show the dataset's attributes and their range.
        histogram()
            Generate a histogram that displays the dictionary.
        compare(other dataset object)
            Generates a plot to compare two different datasets/categories by the relative proportion of each label.
    """

    def __init__(self, path, print_info=True):
        if print_info:
            print("Using data from:", path)

        self.path = path
        self.name = self.path.split('/')[-1]
        self.data = self.load_first(self.path)
        self.numsamples = len(self.data)
        self.numattributes= len(self.data[0]) - 1
        self.labels = self.get_labels(self.data)
        self.attributes = self.get_attributes(self.data)
        self.dictionary = self.set_up_count_dict(self.labels)

    def load_first(self, file):
        return np.loadtxt(file, dtype=str, delimiter=',')

    def get_labels(self, ar):
        return ar[:, np.size(ar, 1) - 1]

    def get_attributes(self, ar):
        return ar[:, :np.size(ar, 1) - 1].astype(int)

    def set_up_count_dict(self, labels):
        counter = {}
        for key in np.unique(labels):
            counter.update({key: np.count_nonzero(labels == key)})
        return counter

    def att_range(self, att_no):
        return (np.amin(self.attributes[:, att_no]), np.amax(self.attributes[:, att_no]))

    def range_graph(self):
        low = []
        high = []
        attributes = []
        for att_num in range(0, self.numattributes):
            attributes.append(att_num)
            att_range = self.att_range(att_num)
            low.append(att_range[0])
            high.append(att_range[1])
        print(low)
        print(high)
        plt.plot((attributes,attributes), (low, high), '--*')
        plt.xlabel('Attribute Number')
        plt.ylabel('Attribute Value Range')
        plt.title('Dataset Attribute Ranges')
        plt.legend(attributes, loc='right',bbox_to_anchor=(1.11, 0.5))
        for val in attributes:
            plt.annotate(high[val]-low[val],xy=(val, high[val]))
        plt.show()
    def range_graph2(self):
        box_array = []
        attributes = []
        low = []
        high = []
        for att_num in range(0, self.numattributes):
            attributes.append(att_num)
            att_range = self.att_range(att_num)
            box_array.append(self.attributes[:, att_num])
            low.append(att_range[0])
            high.append(att_range[1])

        bp = plt.boxplot(box_array, 0, 'rD', showcaps=True, showfliers=True, showmeans=True)
        plt.setp(bp['boxes'], color='black')
        plt.setp(bp['whiskers'], color='black')
        plt.setp(bp['fliers'], color='red', marker='+')
        plt.xticks(np.add(attributes, 1), att_name.values(), rotation=45)
        for val in attributes:
            plt.annotate(high[val]-low[val],xy=(val + 1, high[val] ), color='b')
        plt.xlabel('Attribute Name')
        plt.ylabel('Attribute Value Range')
        plt.title('Dataset Attribute Ranges')
        plt.show()

    def histogram(self):
        plt.bar(self.dictionary.keys(), self.dictionary.values(), color='g')
        plt.show()

    def subcategorybar(self, X, vals, width=0.8):
        n = len(vals)
        _X = np.arange(len(X))
        for i in range(n):
            plt.bar(_X - width / 2. + i / float(n) * width, vals[i],
                    width=width / float(n), align="edge")
            count = 0
            for val in vals[i]:
                plt.annotate(np.round(val,1), xy=(_X[count] - width / 2. + i / float(n) * width, val))
                count+=1
        plt.xticks(_X, X)

    def compare(self, other):
        newdict = {}
        ourdict = {}
        otherdict = {}
        for key1 in self.dictionary.keys():
            newdict[key1] = self.dictionary[key1]/self.numsamples*100
            ourdict[key1] = newdict[key1]
            otherdict[key1] = 0

        for key2 in other.dictionary.keys():
            if key2 in newdict.keys():
                newdict[key2] = newdict[key2] - other.dictionary[key2]/other.numsamples*100
                otherdict[key2] = other.dictionary[key2]/other.numsamples*100
            else:
                newdict[key2] = -other.dictionary[key2]/other.numsamples*100
                otherdict[key2] = -newdict[key2]
                ourdict[key2] = 0

        self.subcategorybar(newdict.keys(), [ourdict.values(), otherdict.values(), newdict.values()])
        plt.xlabel('Category')
        plt.ylabel('Proportion [%]')
        plt.title('Dataset Proportional Comparison')
        plt.legend([self.name, other.name, 'Difference'], loc='right')
        plt.show()

if __name__ == "__main__":
    dataset = Dataset('data/train_full.txt')
    dataset2 = Dataset('data/train_sub.txt')
    dataset.range_graph2()
    #dataset.compare(dataset2)
    print(dataset.numsamples)
    print(dataset2.numsamples)



