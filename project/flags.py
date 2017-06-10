import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('Run_Mode', 'dev', '''run mode''')
tf.app.flags.DEFINE_boolean('Verbose_Mode', True, '''Verbose_Mode''')

tf.app.flags.DEFINE_integer('IMG_SIZE', 64, '''IMG_SIZE''')
tf.app.flags.DEFINE_integer('IMG_CHANNEL', 3, '''IMG_CHANNEL''')
tf.app.flags.DEFINE_integer('NUM_CLASS', 200, '''NUM_CLASS''')

tf.app.flags.DEFINE_integer('Train_Batch_Size', 100, '''Train_Batch_Size''')
tf.app.flags.DEFINE_integer('Val_Batch_Size', 512, '''Validation_Batch_Size''')
tf.app.flags.DEFINE_integer('Test_Batch_Size', 200, '''Test_Batch_Size''')
tf.app.flags.DEFINE_integer('Test_Batch_Count', 50, '''Test_Batch_Count''')

tf.app.flags.DEFINE_integer('Train_Steps', 20000, '''Total steps that you want to train''')
tf.app.flags.DEFINE_integer('Learning_Rate', 0.01, '''Learning_Rate''')
tf.app.flags.DEFINE_float('Weight_Decay', 0.0002, '''scale for l2 regularization''')

tf.app.flags.DEFINE_integer('Part_Val_Freq', 10, '''Val_Freq''')
tf.app.flags.DEFINE_integer('Full_Val_Freq', 500, '''Full_Val_Freq''')
tf.app.flags.DEFINE_integer('CKPT_FREQ', 1000, '''Log Frequency''')

tf.app.flags.DEFINE_boolean('USE_CKPT', True, '''USE_CKPT''')
tf.app.flags.DEFINE_integer('CUR_STEP', 39999, '''Current global step''')

tf.app.flags.DEFINE_string('CKPT_PATH', "ckpts/model_ckpt.dat", '''CKPT_PATH''')
tf.app.flags.DEFINE_string('LOG_PATH', "logs/", '''LOG_PATH''')
