import logging
from textwrap import dedent

import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import Conflict, NotFound


class BigqueryWorker:
    def __init__(self, app=None):
        self.logger = app.logger if app else logging.getLogger()
        self.client = bigquery.Client(project='storm-0809', location='asia-northeast3')
        self.dataset_ref = self.client.dataset('stock')

    def get_table_if_exists(self, table_name):
        table_ref = bigquery.TableReference(self.dataset_ref, table_name)
        try:
            table = self.client.get_table(table_ref)
        except NotFound:
            return None
        else:
            return table

    def get_itemcodes_info(self):
        table = self.get_table_if_exists('itemcodes_info')
        df = self.client.list_rows(table).to_dataframe(dtypes=self._extract_dtypes(table))
        return df

    def save_itemcodes_info(self, df, write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE):
        table_name = 'itemcodes_info'
        table_ref = self._save_info('itemcode', table_name, df, write_disposition=write_disposition)

        return table_ref

    def get_daily_item_info(self, itemcode_info, start_date=None, end_date=None):
        return self._get_daily_info('daily_items', itemcode_info, start_date, end_date)

    def save_daily_item_info(self, itemcode_info, df, write_disposition=bigquery.WriteDisposition.WRITE_EMPTY):
        itemcode, itemname, market = itemcode_info
        table_name = f'daily_items_info_{itemcode}_{market}'
        table_ref = self._save_info('daily_items', table_name, df, write_disposition)

        return table_ref

    def get_daily_item_indicator_info(self, itemcode_info, start_date=None, end_date=None):
        return self._get_daily_info('daily_items_indicator', itemcode_info, start_date, end_date)

    def save_daily_item_indicator_info(
            self, itemcode_info, df, write_disposition=bigquery.WriteDisposition.WRITE_EMPTY):
        itemcode, itemname, market = itemcode_info
        table_name = f'daily_items_indicator_info_{itemcode}_{market}'
        table_ref = self._save_info('daily_items_indicator', table_name, df, write_disposition)

        return table_ref

    def get_last_date_of_daily_info(self, table_name, codeinfo_df=None):
        itemnames_clause=''
        if codeinfo_df is not None:
            itemnames_str = ', '.join([f"'{x}'" for x in codeinfo_df['itemname'].unique().tolist()])
            itemnames_clause = f"WHERE itemname in ({itemnames_str})"
        sql = dedent(f"""
            SELECT MAX(date) lastdate
            FROM stock.{table_name}
            {itemnames_clause}
        """)
        df = self.client.query(sql).to_dataframe(dtypes={'lastdate': np.dtype('datetime64[ns]')})
        lastdate = df.lastdate[0]

        return lastdate

    def get_daily_info_all(self, info_type, codeinfo_df=None, start_date=None, end_date=None):
        wheres = []
        if codeinfo_df is not None and not codeinfo_df.empty:
            item_names = ', '.join([f"'{x}'" for x in codeinfo_df['itemname']])
            wheres.append(f"itemname IN ({item_names})")
        if start_date:
            wheres.append(f"date >= '{start_date.format('YYYY-MM-DD')}'")
        if end_date:
            wheres.append(f"date <= '{end_date.format('YYYY-MM-DD')}'")
        where_clause = ' AND '.join(wheres)
        where_clause = 'WHERE ' + where_clause if where_clause else ''
        query = dedent(f"""
            SELECT *
            FROM stock.{info_type}
            {where_clause}
        """)
        table = self.get_table_if_exists(info_type)
        df = self.client.query(query).to_dataframe(dtypes=self._extract_dtypes(table))

        return df

    def save_daily_info_all(self, info_type, codeinfo_df, write_disposition=bigquery.WriteDisposition.WRITE_EMPTY,
                            start_date=None, end_date=None):
        all_table_names = [x.table_id for x in self.client.list_tables(self.dataset_ref)]
        table_names = [f"{info_type}_{data['itemcode']}_{data['market']}" for index, data in codeinfo_df.iterrows()]
        table_names = [x for x in table_names if x in all_table_names]
        where_clause = ' AND '.join([
            f"date {'>=' if i == 0 else '<='} '{x.format('YYYY-MM-DD')}'"
            for i, x in enumerate([start_date, end_date]) if x])
        where_clause = 'WHERE ' + where_clause if where_clause else ''
        query = 'UNION ALL'.join([dedent(f"""
            SELECT *
            FROM stock.{table_name}
            {where_clause}
        """) for table_name in table_names])

        result_table_name = f'{info_type}_all'
        job_config = bigquery.QueryJobConfig(write_disposition=write_disposition)
        table_ref = bigquery.TableReference(self.dataset_ref, result_table_name)
        job_config.destination = table_ref
        job = self.client.query(query, job_config=job_config)
        job.result()

        return table_ref

    def _get_daily_info(self, info_type, itemcode_info, start_date=None, end_date=None):
        itemcode, itemname, market = itemcode_info
        table = self.get_table_if_exists(f'{info_type}_info_{itemcode}_{market}')
        daily_info = self.client.list_rows(table).to_dataframe(dtypes=self._extract_dtypes(table))
        daily_info = daily_info.set_index('date').sort_index()
        daily_info.index = pd.to_datetime(daily_info.index)
        if start_date:
            daily_info = daily_info.loc[start_date:]
        if end_date:
            daily_info = daily_info.loc[:end_date]

        return daily_info

    def _extract_schema(self, df):
        schema = []
        for column_name, dtype in dict(df.dtypes).items():
            type_map = {
                np.dtype('object'): bigquery.SchemaField(column_name, 'STRING'),
                np.dtype('int32'): bigquery.SchemaField(column_name, 'INTEGER'),
                np.dtype('int64'): bigquery.SchemaField(column_name, 'INTEGER'),
                np.dtype('float64'): bigquery.SchemaField(column_name, 'FLOAT'),
                np.dtype('datetime64[ns]'): bigquery.SchemaField(column_name, 'DATE'),
                pd.DatetimeTZDtype(tz='UTC'): bigquery.SchemaField(column_name, 'DATE')
            }
            schema.append(type_map[dtype])

        return schema

    def _extract_dtypes(self, table):
        dtypes = dict()
        type_map = {
            'STRING': np.dtype('object'),
            'INTEGER': np.dtype('int64'),
            'FLOAT': np.dtype('float64'),
            'DATE': np.dtype('datetime64[ns]'),
        }
        for field in table.schema:
            dtypes[field.name] = type_map[field.field_type]
        return dtypes

    def _save_info(self, info_type, table_name, df, write_disposition=bigquery.WriteDisposition.WRITE_EMPTY):
        table_ref = bigquery.TableReference(self.dataset_ref, table_name)
        schema = self._extract_schema(df)
        try:
            table = self.client.get_table(table_ref)
        except NotFound:
            table = bigquery.Table(table_ref, schema=schema)
            table = self.client.create_table(table)
        job_config = bigquery.LoadJobConfig(
            schema=schema, write_disposition=write_disposition
        )
        try:
            job = self.client.load_table_from_dataframe(df, table, job_config=job_config)
        except Conflict:
            self.logger.info(f'{table.table_id} 테이블이 이미 존재하여 SKIP')
            return
        job.result()
        self.logger.info(f'{table.table_id} 테이블에 {info_type} 정보 저장')
        if 'date' in df:
            start_date, end_date = df['date'].min(), df['date'].max()
            self.logger.info(f'start_date: {start_date}, end_date: {end_date}')

        return table

    def delete_duplicated_rows(self, table_name, start_date_str, codeinfo_df=None):
        if not self.get_table_if_exists(table_name):
            return

        itemnames_clause=''
        if codeinfo_df is not None:
            itemnames_str = ', '.join([f"'{x}'" for x in codeinfo_df['itemname'].unique().tolist()])
            itemnames_clause = f"AND itemname in ({itemnames_str})"
        job = self.client.query(dedent(f"""
            DELETE
            FROM stock.{table_name}
            WHERE date >= '{start_date_str}'
                {itemnames_clause}
        """))
        job.result()
        self.logger.info(f'{table_name} 테이블에서 {start_date_str} 이후 데이터 삭제 ({codeinfo_df})')
