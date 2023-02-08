# coding=utf-8
import pandas as pd
import numpy as np
import pickle


df_colnames_default = {
    "Название продукта": "name",
    "Описание продукта": "descr",
    "Код ОКПД2": "okpd2",
    "Название ОКПД2": "okpd2_name",
    "Код КТРУ": "ktru",
    "Название КТРУ": "ktru_name"
}


def read_csv(path_to_csv: str,
             index_col: str,
             colnames: dict = None) -> pd.DataFrame:
    df = pd.read_csv(path_to_csv, index_col=index_col)
    if colnames is not None:
        df = df.rename(columns=colnames)
    df = df.drop_duplicates()
    return df


def code_filter(code) -> int or None:
    code = str(code)
    if code[-2:] == '.0':
        code = code[:-2]
    if not code.isnumeric():
        return None
    return int(code)


def concat_strs_from_list(lst: list) -> str:
    res = ""
    for s in lst:
        res += s + ' '
    return res


class DataframeBuilder:
    def __init__(self, *,
                 path_to_data_csv: str = "data/products_with_ktru.csv",
                 index_col: str = "id",
                 colnames: dict = None,
                 target: str = "ktru_name",
                 class_balance_type: str = "balanced",
                 balanced_min_examples_per_class: int = 20,
                 balanced_max_examples_per_class: int = 200,
                 stratified_num_of_examples: int = 25000,
                 stratified_drop_threshold: int = 4):
        """
        :param path_to_data_csv: path to the table with data
        :param index_col: index column name of the table with data
        :param colnames: dict to rename the columns (must be as in df_colnames_default)
        :param target: name of the target column
        :param class_balance_type: "balanced" or "stratified"
        :param balanced_min_examples_per_class: minimum number of examples of a class to be included to the resulted df
                                                (only for class_balance_type="balanced")
        :param balanced_max_examples_per_class: maximum number of -//-
        :param stratified_num_of_examples: number of examples in the resulted df
                                            (only for class_balance_type="stratified")
        :param stratified_drop_threshold: if after stratification, there are less than stratified_drop_threshold
                                        examples for some class, drop this class from the resulted df.
                                        Recommended value is not less than 4 due to the further train_test_split stratification
                                        (only for class_balance_type="stratified")
        """
        self.df = None
        self.path_to_data_csv = path_to_data_csv
        self.index_col = index_col
        if colnames is None:
            self.colnames = df_colnames_default
        else:
            self.colnames = colnames
        self.target = target
        self.class_balance_type = class_balance_type
        self.balanced_min_examples_per_class = balanced_min_examples_per_class
        self.balanced_max_examples_per_class = balanced_max_examples_per_class
        self.stratified_num_of_examples = stratified_num_of_examples
        self.stratified_drop_threshold = stratified_drop_threshold

    def build(self, output_path: str = "data/df.csv"):
        self.df = read_csv(self.path_to_data_csv, self.index_col, self.colnames)
        self.df = self.add_features(self.df)

        if self.class_balance_type == "stratified":
            self.df = self.stratify(self.df, self.stratified_num_of_examples, self.stratified_drop_threshold)
        else:
            self.df = self.balance(self.df, self.balanced_min_examples_per_class, self.balanced_max_examples_per_class)

        self.add_okpd2_names(self.df)

        self.df.to_csv(output_path)

    def build_nonlabeled(self, *, ids: pd.Series = None, num_of_examples: int = 100, output_path: str = "data/df_nonlabaled.csv"):
        df_products = read_csv("data/products.csv", index_col="id")
        df_products = df_products[df_products['ktru'].isna()]
        df_products = df_products.loc[:, ~df_products.columns.str.contains('^Unnamed')]
        df_products = df_products[['product_name', "product_descr", "okpd2_code", "okpd2_name", "okei_code"]]
        colnames = {
            "product_name": "name",
            "product_descr": "descr",
            "okpd2_code": "okpd2",
            "okei_code": "okei"
        }
        df_products = df_products.rename(columns=colnames)
        df_products = df_products.drop_duplicates()
        if ids is None:
            df_products = df_products.sample(n=num_of_examples)
        else:
            df_products = df_products.loc[ids]

        df_products['okei'].fillna("", inplace=True)
        df_products['okei'] = df_products['okei'].apply(code_filter)

        df_products = self.add_companies(df_products)
        df_products = self.add_keywords(df_products)
        df_products = self.add_tnved_code(df_products)
        df_products = self.add_okpd2_names(df_products)

        df_products.to_csv(output_path)


    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_okei_code(df)
        df = self.add_companies(df)
        df = self.add_keywords(df)
        df = self.add_tnved_code(df)
        return df

    def add_okei_code(self, df: pd.DataFrame) -> pd.DataFrame:
        df_products = read_csv("data/products.csv", index_col="id")
        df_join = df.join(df_products, rsuffix='_right')
        df['okei'] = ""
        df['okei'] = df_join['okei_code'].dropna()
        df['okei'] = df['okei'].fillna("")
        df['okei'] = df['okei'].apply(code_filter)
        df['okei'] = df['okei'].apply(str)
        return df

    def add_companies(self, df: pd.DataFrame) -> pd.DataFrame:
        df_prod_comp = pd.read_csv("data/products_companies.csv", index_col='product_id')
        df_prod_comp = df_prod_comp['company_id']
        df = df.join(df_prod_comp)

        df_companies = pd.read_csv("data/companies.csv", index_col='id', dtype=str)
        df_companies = df_companies[['name', 'full_name', 'okved2_id']]
        colnames = {
            "id": "company_id",
            "name": "company_name",
            "full_name": "company_fullname"
        }
        df_companies = df_companies.rename(columns=colnames)

        df_companies['okved2_id'] = df_companies['okved2_id'].fillna("")
        df_companies['okved2_id'] = df_companies['okved2_id'].apply(code_filter)

        df_okved2 = pd.read_csv("data/okved2.csv", index_col='id', dtype=str)
        df_okved2 = df_okved2[['name']]
        colnames = {
            "id": 'okved2_id',
            "name": "okved2_name"
        }
        df_okved2 = df_okved2.rename(columns=colnames)

        df_companies = df_companies.join(df_okved2, on='okved2_id')
        df = df.join(df_companies, on='company_id')
        return df

    def add_keywords(self, df: pd.DataFrame) -> pd.DataFrame:
        df_keywords = pd.read_csv("data/products_keywords.csv", index_col='product_id')
        df = df.join(df_keywords)
        return df

    def add_tnved_code(self, df: pd.DataFrame) -> pd.DataFrame:
        df_prod_tnved = pd.read_csv("data/product_tnved.csv", index_col='tnved_id')
        df_tnved = pd.read_csv("data/tnved.csv", index_col='id')
        df_tnved = df_tnved[['name']]
        df_prod_tnved = df_prod_tnved.join(df_tnved)
        df_prod_tnved = df_prod_tnved[['product_id', 'name']]
        df_prod_tnved = df_prod_tnved.set_index('product_id')
        df_prod_tnved = df_prod_tnved.rename(columns={'name': 'tnved_name'})
        df_prod_tnved.index.rename('', inplace=True)
        df = df.join(df_prod_tnved)
        return df

    def balance(self, df: pd.DataFrame, min_examples_per_class: int, max_examples_per_class: int) -> pd.DataFrame:
        classes_to_take = df[self.target].value_counts()[df[self.target].value_counts() >= min_examples_per_class]
        num_of_classes = len(classes_to_take)
        df_balanced = pd.DataFrame(columns=df.columns)
        for ktru_target in classes_to_take[0:num_of_classes].index:
            df_ktru = df[df[self.target] == ktru_target]
            num = max_examples_per_class if len(df_ktru) > max_examples_per_class else len(df_ktru)
            df_ktru = df_ktru.sample(num)
            df_balanced = pd.concat([df_balanced, df_ktru])
        return df_balanced

    def stratify(self, df: pd.DataFrame, num_of_examples: int, stratified_drop_threshold: int) -> pd.DataFrame:
        # stratify the dataframe according to the target
        init_shares = df[self.target].value_counts(normalize=True)
        df_balancer = df.reset_index().set_index(self.target)
        df_strat = df_balancer.sample(n=num_of_examples, weights=init_shares)
        df_strat = df_strat.reset_index().set_index('index')
        df = df_strat
        # drop classes that have less than stratified_drop_threshold examples
        target_freq = df[self.target].value_counts().reset_index()
        target_take = target_freq[target_freq[self.target] >= stratified_drop_threshold]['index']
        df = df[df[self.target].isin(target_take)]
        return df

    def add_okpd2_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df_okpd2 = pd.read_csv('data/okpd2.csv', index_col='code', dtype=str)
        df_okpd2 = df_okpd2[['id', 'name', 'parent_id']]

        def okpd2_code_to_id(code):
            try:
                return str(df_okpd2.loc[code]['id'])
            except KeyError:
                return code

        df['okpd2_id'] = df['okpd2']
        df['okpd2_id'] = df['okpd2_id'].apply(okpd2_code_to_id)

        df['okpd2_names'] = np.empty((len(df), 0)).tolist()

        with open('data/okpd2_dict.pickle', 'rb') as handle:
            okpd2_dict = pickle.load(handle)

            def fill_okpd2_names(row):
                ret_list = []
                id = row['okpd2_id']
                while id != float('nan'):
                    try:
                        okpd2_entry = okpd2_dict[id]
                    except KeyError:
                        break
                    new_name = okpd2_entry[0]
                    id = okpd2_entry[1]
                    ret_list.append(new_name)

                return ret_list

            df['okpd2_names'] = df.apply(fill_okpd2_names, axis=1)

        df['okpd2_names'] = df['okpd2_names'].apply(concat_strs_from_list)

        return df


if __name__ == '__main__':
    dfb = DataframeBuilder()
    #dfb = DataframeBuilder(class_balance_type="stratified", stratified_num_of_examples=150000)
    dfb.build()