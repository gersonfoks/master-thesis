import json

import pyarrow as pa
import pandas as pd


def save_arrow_file(table, ref):
    with pa.OSFile(ref, 'wb') as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)


def load_arrow_file(ref):
    source = pa.memory_map(ref, 'r')
    table = pa.ipc.RecordBatchFileReader(source).read_all()

    return table


def load_arrow_file_in_memory(ref):
    with pa.OSFile(ref, 'rb') as source:
        table = pa.ipc.open_file(source).read_all()
    return table


def save_dict_as_json(dict, ref):
    with open(ref, 'w') as fp:
        json.dump(dict, fp)


def load_json_as_dict(ref):
    with open(ref) as json_file:
        data = json.load(json_file)
    return data


def save_dict_as_csv(dict, ref):
    df = pd.DataFrame.from_dict(dict)
    df.to_csv(ref, sep='\t', index=False)


def load_csv_as_dict(ref):
    return pd.read_csv(ref, sep='\t').to_dict()


def load_csv_as_df(ref):
    return pd.read_csv(ref, sep='\t')
