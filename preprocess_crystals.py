import os
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from pymatgen.core.structure import Structure
from sklearn.model_selection import train_test_split
import pickle

import nfp

from nfp_extensions import CifPreprocessor
tqdm.pandas()

      
if __name__ == '__main__':
        
    # Read energy data
    data = pd.read_csv('/projects/rlmolecule/pgorai/nrelmatdb_icsd.csv')
    data = data.sort_values('energyperatom').drop_duplicates(subset='icsdnum', keep='first').sample(frac=1., random_state=1)

    # So pymatgen doesn't want to take the ISO-8859-1 cifs in the tarball, I have to 
    # re-encode as utf-8 using the following command:
    # for file in *; do iconv -f ISO-8859-1 -t UTF-8 < $file > "../utf8_cifs/$file"; done
    cif_file = lambda x: '/scratch/pstjohn/utf8_cifs/icsd_{:06d}.cif'.format(x)
    cif_exists = lambda x: os.path.exists(cif_file(x))
    data['cif_exists'] = data.icsdnum.apply(cif_exists)
    data = data[data.cif_exists]

    # Drop ICSDs with fractional occupation
    to_drop = pd.read_csv('icsd_fracoccupation.csv', header=None)[0]\
        .str.extract('_(\d{6}).cif').astype(int)[0]
    data = data[~data.icsdnum.isin(to_drop)]

    # Try to parse crystals with pymatgen
    def get_crystal(icsd):
        try:
            return Structure.from_file(cif_file(icsd), primitive=True)
        except Exception:
            return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data['crystal'] = data.icsdnum.progress_apply(get_crystal)

    # record parse issues
    data[data.crystal.isna()]['icsdnum'].to_csv('problems.csv')
        
    data = data.dropna(subset=['crystal'])
    print(f'{len(data)} crystals after down-selection')


    # Split the data into training and test sets
    train, test = train_test_split(data.icsdnum.unique(), test_size=500, random_state=1)
    train, valid = train_test_split(train, test_size=500, random_state=1)
    
    
    # Initialize the preprocessor class.
    preprocessor = CifPreprocessor(num_neighbors=12)

    def inputs_generator(df, train=True):
        """ This code encodes the preprocessor output (and prediction target) in a 
        tf.Example format that we can use with the tfrecords file format. This just
        allows a more straightforward integration with the tf.data API, and allows us
        to iterate over the entire training set to assign site tokens.
        """
        for i, row in tqdm(df.iterrows(), total=len(df)):
            input_dict = preprocessor.construct_feature_matrices(row.crystal, train=train)
            input_dict['energyperatom'] = float(row.energyperatom)

            features = {key: nfp.serialize_value(val) for key, val in input_dict.items()}
            example_proto = tf.train.Example(features=tf.train.Features(feature=features))

            yield example_proto.SerializeToString()

    # Process the training data, and write the resulting outputs to a tfrecord file
    serialized_train_dataset = tf.data.Dataset.from_generator(
        lambda: inputs_generator(data[data.icsdnum.isin(train)], train=True),
        output_types=tf.string, output_shapes=())

    filename = 'tfrecords/train.tfrecord.gz'
    writer = tf.data.experimental.TFRecordWriter(filename, compression_type='GZIP')
    writer.write(serialized_train_dataset)
    
    with open('tfrecords/preprocessor.p', 'wb') as f:
        pickle.dump(preprocessor, f)    

    # Process the validation data
    serialized_valid_dataset = tf.data.Dataset.from_generator(
        lambda: inputs_generator(data[data.icsdnum.isin(valid)], train=False),
        output_types=tf.string, output_shapes=())

    filename = 'tfrecords/valid.tfrecord.gz'
    writer = tf.data.experimental.TFRecordWriter(filename, compression_type='GZIP')
    writer.write(serialized_valid_dataset)
    
    # Save train, valid, and test datasets
    data[data.icsdnum.isin(train)][
        ['icsdnum', 'numatom', 'initialspacegroupnum', 'energyperatom']].to_csv(
        'train.csv.gz', compression='gzip', index=False)
    data[data.icsdnum.isin(valid)][
        ['icsdnum', 'numatom', 'initialspacegroupnum', 'energyperatom']].to_csv(
        'valid.csv.gz', compression='gzip', index=False)
    data[data.icsdnum.isin(test)][
        ['icsdnum', 'numatom', 'initialspacegroupnum', 'energyperatom']].to_csv(
        'test.csv.gz', compression='gzip', index=False)
