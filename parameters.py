

ACTIVE_LEARNING = True
MACHINE_LEARNING = False


REFAEL_PARAM = {
            'logger_name': "logger",
            # ----------------------- Data parameters
            'days_split': 1,
            'start_time': 10,
            'window_size': None,
            'database': 'Refael',
            'data_file_name': 'Refael_07_18.csv',  # should be in ../data/
            'date_format': "%Y-%m-%d",  # Refael
            'directed': True,
            'white_label': 1,
            # graph_measures + beta vectors parameters
            'max_connected': False,

            # ---------------------- ML- parameters
            'min_nodes': 10,
            # 'learn_method': "nn",
            # 'learn_method': "XG_Boost",
            'learn_method': "rand",

            # ---------------------- AL - parameters
            'queries_per_time': 1,
            'batch_size': 8,
            'eps': 0.01,
            'target_recall': 0.7,
            'reveal_target': 0.6,
            # 'ml_method': "nn"
            # 'ml_method': "XG_Boost"
            # 'ml_method': "rand"
        }