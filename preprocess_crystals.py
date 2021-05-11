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
    icsd = pd.read_csv('/projects/rlmolecule/shubham/cgcnn/crystal-gnn/matdb_data/clean_data/batteries/nrel_16488.csv')
    sp   = pd.read_csv('/projects/rlmolecule/shubham/cgcnn/crystal-gnn/matdb_data/clean_data/batteries/relaxed_structures_sp.csv')
    pg   = pd.read_csv('/projects/rlmolecule/shubham/cgcnn/crystal-gnn/matdb_data/clean_data/batteries/relaxed_structures_pg.csv')

    
    # So pymatgen doesn't want to take the ISO-8859-1 cifs in the tarball, I have to 
    # re-encode as utf-8 using the following command:
    # for file in *; do iconv -f ISO-8859-1 -t UTF-8 < $file > "../utf8_cifs/$file"; done

    # path to ICSD cifs files, shubham structures (sp), prahsun structures (pg)
    cif_icsd = lambda x: '/projects/rlmolecule/shubham/cgcnn/crystal-gnn/matdb_data/clean_data/nrel_jiaxing/utf8_cifs/{}.cif'.format(x)
    cif_sp = lambda x,y,z: '/scratch/shubhampandey/ARPA-E/battery_decorations/relaxed/{}/{}/{}/CONTCAR'.format(x,y,z)
    cif_pg = lambda x,y,z: '/scratch/pgorai/decoration-relax/relaxed/{}/{}/{}/CONTCAR'.format(x,y,z)


    # check if structures corresponding to all 'id' exists
    icsd_exists = lambda x: os.path.exists(cif_icsd(x))
    icsd['cif_exists'] = icsd.id.apply(icsd_exists)
    icsd = icsd[icsd.cif_exists]
         
    sp_exists = lambda x,y,z: os.path.exists(cif_sp(x,y,z))
    sp['cif_exists'] = sp[['comp_type','composition','id']].apply(lambda row: sp_exists(row['comp_type'], row['composition'],row['id']) , axis=1)
    sp = sp[sp.cif_exists]

    pg_exists = lambda x,y,z: os.path.exists(cif_pg(x,y,z))
    pg['cif_exists'] = pg[['comp_type','composition','id']].apply(lambda row: pg_exists(row['comp_type'], row['composition'],row['id']) , axis=1)
    pg = pg[pg.cif_exists]

    
    # Drop ICSDs with fractional occupation
    #to_drop = pd.read_csv('icsd_fracoccupation.csv', header=None)[0]\
        #.str.extract('_(\d{6}).cif').astype(int)[0]
    #data = data[~data.icsdnum.isin(to_drop)]

    # Try to parse ICSD crystals with pymatgen
    def get_icsd(id):
        try:
            return Structure.from_file(cif_icsd(id), primitive=True)
        except Exception:
            return None

    # Try to parse sp crystals with pymatgen
    def get_sp(ctype,comp,str_id):
        try:
            return Structure.from_file(cif_sp(ctype,comp,str_id), primitive=True)
        except Exception:
            return None


    # Try to parse pg crystals with pymatgen
    def get_pg(ctype,comp,str_id):
        try:
            return Structure.from_file(cif_pg(ctype,comp,str_id), primitive=True)
        except Exception:
            return None

    
    # record ICSD structures as a column
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        icsd['crystal'] = icsd.id.progress_apply(get_icsd)


    # record sp structures as a column
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp['crystal'] = sp[['comp_type','composition','id']].progress_apply(lambda row: get_sp(row['comp_type'], row['composition'],row['id']) , axis=1)


    # record pg structures as a column
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pg['crystal'] = pg[['comp_type','composition','id']].progress_apply(lambda row: get_pg(row['comp_type'], row['composition'],row['id']) , axis=1)
    

    # record parse issues
    icsd[icsd.crystal.isna()]['id'].to_csv('problems_icsd.csv')
        
    # record parse issues
    sp[sp.crystal.isna()]['id'].to_csv('problems_sp.csv')

    # record parse issues
    pg[pg.crystal.isna()]['id'].to_csv('problems_pg.csv')


    icsd = icsd.dropna(subset=['crystal'])
    sp   = sp.dropna(subset=['crystal'])
    pg   = pg.dropna(subset=['crystal'])
    print(f'{len(icsd),len(sp),len(pg)} icsd,sp,pg crystals after down-selection')
  
    # combine sp and pg
    hypo = sp.append(pg).reset_index(drop=True)
    
    # Split the icsd data into training and test sets
    train_icsd, test_icsd  = train_test_split(icsd.id.unique(), test_size=500, random_state=1)
    train_icsd, valid_icsd = train_test_split(train_icsd, test_size=500, random_state=1)

    '''
    # Split the hypothetical data into training and test sets
    train_hypo, valid_hypo = train_test_split(hypo.composition.unique(), test_size=20, random_state=1)
    train_hypo, test_hypo  = train_test_split(train_hypo, test_size=20, random_state=1)
    '''  
    
    # Split the hypothetical data into training and test sets, such that test/valid sets have one composition per comp_type
    valid_hypo = hypo.groupby("comp_type").sample(n=1, random_state=1)    
    train_hypo = hypo[~hypo.composition.isin(valid_hypo.composition)]
    test_hypo  = train_hypo.groupby("comp_type").sample(n=1, random_state=1)
    train_hypo = train_hypo[~train_hypo.composition.isin(test_hypo.composition)]

    # merge train, valid, test sets from icsd and hypo data above
    icsd_train = icsd[icsd.id.isin(train_icsd)]
    train      = icsd_train.append(train_hypo)

    icsd_valid = icsd[icsd.id.isin(valid_icsd)]
    hypo_valid = hypo[hypo.composition.isin(valid_hypo.composition)]
    valid      = icsd_valid.append(hypo_valid)

    icsd_test  = icsd[icsd.id.isin(test_icsd)]
    hypo_test  = hypo[hypo.composition.isin(test_hypo.composition)]
    test       = icsd_test.append(hypo_test)


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
        lambda: inputs_generator(train, train=True),
        output_types=tf.string, output_shapes=())

    os.mkdir('tfrecords_nrel500_hypo58')
    dir = 'tfrecords_nrel500_hypo58'
     
    filename = dir + '/train.tfrecord.gz'
    writer = tf.data.experimental.TFRecordWriter(filename, compression_type='GZIP')
    writer.write(serialized_train_dataset)
    
    # Save the preprocessor data
    preprocessor.to_json(dir + '/preprocessor.json')

    # Process the validation data
    serialized_valid_dataset = tf.data.Dataset.from_generator(
        lambda: inputs_generator(valid, train=False),
        output_types=tf.string, output_shapes=())

    filename = dir + '/valid.tfrecord.gz'
    writer = tf.data.experimental.TFRecordWriter(filename, compression_type='GZIP')
    writer.write(serialized_valid_dataset)
    
    # Save train, valid, and test datasets
    train[
        ['comp_type','composition', 'id', 'energyperatom']].to_csv(
        dir+'/train.csv.gz', compression='gzip', index=False)
    valid[
        ['comp_type','composition', 'id', 'energyperatom']].to_csv(
        dir+'/valid.csv.gz', compression='gzip', index=False)
    test[
        ['comp_type','composition', 'id', 'energyperatom']].to_csv(
        dir+'/test.csv.gz', compression='gzip', index=False)
