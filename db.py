
import psycopg2

import os
from dotenv import load_dotenv
load_dotenv()


class Database():
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self):
        # connection = psycopg2.connect(database=os.getenv("DATABASE"), user=os.getenv("USER"), password=os.getenv("PASSWORD"), host=os.getenv("HOST"), port=os.getenv("PORT"))
        connection = psycopg2.connect(
            database="eidetic",
            user="postgres",
            password="banana",
            host="localhost",
            port=5432
        )

        connection.autocommit = True
        self.connection = connection

    def insert_record(self, record, table_number):

        record_str = ""
        i = 1
        for val in record:
            if i != len(record):
                record_str = record_str + "(" + str(i) +", " + str(val) + "),"
            else: 
                record_str = record_str + "(" + str(i) +", " + str(val) + ");"
            i = i + 1
            
        
        cursor = self.connection.cursor()

        cursor.execute("INSERT INTO percentile_activations_" + str(table_number) + "(node_id, activation) values " + record_str)

    def recreate_tables(self, num_quantiles, table_number):

        if num_quantiles <= 1:
            return
            
        cursor = self.connection.cursor()

        sql_1 = '''create table percentile_activations_''' + str(table_number) + '''(
            node_id		int not null,
            activation	double precision not null
        );'''

        sql_2 = '''CREATE INDEX node_idx_''' + str(table_number) + ''' ON percentile_activations_'''+ str(table_number) +''' (node_id);'''
        

        extended_sql_3 = ""

        for i in range(1, num_quantiles):

            if i != num_quantiles -1:
                extended_sql_3 = extended_sql_3 + "threshold_" + str(i) + " double precision not null,\n"
            else: 
                extended_sql_3 = extended_sql_3 + "threshold_" + str(i) + " double precision not null\n"

        sql_3 = '''create table percentile_distribution_''' + str(table_number) + ''' (
            node_id		int not null,\n''' + extended_sql_3 + '''
            
        );'''

        sql_4 = '''CREATE INDEX node_idx_dist_''' + str(table_number) + ''' ON percentile_distribution_''' + str(table_number) + '''(node_id);'''

        sql_5 = '''drop table if exists percentile_activations_''' + str(table_number) + ''';'''

        sql_6 = '''drop table if exists percentile_distribution_''' + str(table_number) + ''';'''

        cursor.execute(sql_5)
        cursor.execute(sql_6)
        cursor.execute(sql_1)
        cursor.execute(sql_2)
        cursor.execute(sql_3)
        cursor.execute(sql_4)

    def create_quantile_distribution(self, num_quantiles, table_number):
        cursor = self.connection.cursor()

        extended_query = ""

        for i in range(1, num_quantiles):
            val = i / float((num_quantiles))
            if i != num_quantiles -1:
                extended_query = extended_query + "percentile_disc(" + str(val) + ") within group (order by percentile_activations_" + str(table_number) + ".activation),\n"
            else:
                extended_query = extended_query + "percentile_disc(" + str(val) + ") within group (order by percentile_activations_" + str(table_number) + ".activation)\n"

        query = '''insert into percentile_distribution_''' + str(table_number) + ''' \n select node_id,
        ''' + extended_query + '''
        from percentile_activations_''' + str(table_number) + '''
            group by node_id
            order by node_id;'''
        
        cursor.execute(query)
        
        query_2 = "select * from percentile_distribution_" + str(table_number)
        cursor.execute(query_2)
        rows = cursor.fetchall()
        
        return rows
        
if os.getenv("USE_DB") == "True":
    database = Database()