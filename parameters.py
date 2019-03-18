

ACTIVE_LEARNING = True
MACHINE_LEARNING = False


REFAEL_PARAM = {
            'logger_name': "logger",
            # ----------------------- Data parameters
            'days_split': 1,
            'start_time': 0,
            'window_size': None,
            'database': 'Refael_12_18',
            'data_file_name': 'Refael_18_12_18_Binary.csv',  # should be in ../data/
            # 'data_file_name': 'Refael_18_12_18_Color_2.csv',
            'date_format': "%Y-%m-%d",  # Refael
            'directed': True,
            'white_label': 0,
            # graph_measures + beta vectors parameters
            'max_connected': False,
            'ftr_pairs': 75,

            # ---------------------- ML- parameters
            'min_nodes': 3,
            # 'min_nodes': 0,
            # 'learn_method': "nn",
            'learn_method': "XG_Boost",
            # 'learn_method': "rand",
            # 'task': 'Multiclass2',  # Multiclass1, Multiclass2, or Binary
            'task': 'Binary',

            # ---------------------- AL - parameters
            'queries_per_time': 1,
            'batch_size': 15,
            'eps': 0.01,
            'target_recall': 0.7,
            'reveal_target': 0.6,
            # 'ml_method': "nn"
            # 'ml_method': "XG_Boost"
            # 'ml_method': "rand"
        }
