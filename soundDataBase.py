from glob import glob

class SoundDataBase(object):
    def __init__(self,db_name,db_src,db_file_type):
        self.db_name = db_name
        self.db_file_type = db_file_type
        self.db_src = db_src


class EscSoundDataBase(SoundDataBase):
    def __init__(self,db_name,db_src,db_file_type,cross_validate = True):
        super(EscSoundDataBase,self).__init__(db_name,db_src,db_file_type)
        self.classes = self.get_classes()
        self.train_files = None
        self.test_files = None
        # self.feature_columns = ['FileName','ClassLabel','Minimum','Maximum','Mean','Median','Variance','Energy','Entropy','TenPentile','NintyPercentile',
        #                            'InterQuartileRange','Range','MeanAbsoluteDeviation','RobustMeanAbsoluteDeviation','RootMeanSquareError',
        #                            'Skewness','Kurtosis','CentroidY','Roundness','Flatness']

    def get_classes(self):
        classes = []
        classes_paths = glob(self.db_src+"**")
        #if not classes_paths: print('Empty classes_paths')        
        #print('get classes')        
        for class_path in classes_paths:
            classes.append(class_path.split('\\')[-1])
            #print(class_path)
        print(classes)
        return classes


    def get_files(self,cls):
        files = glob(cls + "/*.{}".format(self.db_file_type))
        #print(cls + "/*.{}".format(self.db_file_type))
        #if not files: print('no files')
        #else: print(files)
        return files


