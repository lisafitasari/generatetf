import kfp
from kfp import components
from kfp import dsl
from kfp import gcp

class ObjectDict(dict):
  def __getattr__(self, name):
    if name in self:
      return self[name]
    else:
      raise AttributeError("No such attribute: " + name)


@dsl.pipeline(
   name='TF training and prediction pipeline',
   description=''
)
def kubeflow_training(
    output='', 
    project='',
    evaluation='gs://dataset-image-train/TFRecords/images/test_labels.csv',
    train='gs://dataset-image-train/TFRecords/images/train_labels.csv',
    schema='gs://ml-pipeline-playground/flower/schema.json',
    learning_rate=0.1,
    hidden_layer_size='100,50',
    steps=2000,
    target='label',
    workers=0,
    pss=0,
    preprocess_mode='local',
    predict_mode='local',
    optimizer_choice='SGD',
    batch_size_predict='',
    lambda_target=''
):
  output_template = str(output) + '/{{workflow.uid}}/{{pod.name}}/data'
  start_step = 1
  use_gpu = False

  
  if start_step <= 1:
    preprocess = dsl.ContainerOp(
      name='preprocess',
      image='gcr.io/celerates-playground/dock-img:latest',
      arguments=[
          '--training_data_file_pattern', train,
          '--evaluation_data_file_pattern', evaluation,
          '--schema', schema,
          '--gcp_project', project,
          '--run_mode', preprocess_mode,
          '--preprocessing_module', '',
          '--transformed_data_dir', output_template],
      file_outputs={'transformed_data_dir': '/output.txt'}
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))
	
  else:
    preprocess = ObjectDict({
      'outputs': {
        'transformed_data_dir': output_template
      }
    }).apply(gcp.use_gcp_secret('user-gcp-sa'))

  
  if start_step <= 2:
    training = dsl.ContainerOp(
      name='training',
      image='gcr.io/celerates-playground/ml-pipeline-kubeflow-tf-trainer:latest',
        arguments=[
          '--transformed_data_dir', preprocess.output,
          '--schema', schema,
          '--learning_rate', learning_rate,
          '--hidden_layer_size', hidden_layer_size,
          '--steps', steps,
          '--target', target,
          '--preprocessing_module', '',
          '--optimizer', optimizer_choice,
          '--training_output_dir', output_template],
      file_outputs={'training_output_dir': '/output.txt'}
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))
  else:
    training = ObjectDict({
      'outputs': {
        'training_output_dir': output_template
      }
    }).apply(gcp.use_gcp_secret('user-gcp-sa'))
	
  if use_gpu:
        training.image = 'gcr.io/ml-pipeline/ml-pipeline-kubeflow-tf-trainer-gpu:fe639f41661d8e17fcda64ff8242127620b80ba0',
        training.set_gpu_limit(1)

  if start_step <= 3:
    prediction = dsl.ContainerOp(
       name='prediction',
        image='gcr.io/celerates-playground/ml-pipeline-dataflow-tf-predict:latest',
        arguments=[
          '--data_file_pattern', evaluation,
          '--schema', schema,
          '--target_column', target,
          '--model', training.output,
          '--run_mode', predict_mode,
          '--gcp_project', project,
          '--batchsize', batch_size_predict,
          '--predictions_dir', output_template],
      file_outputs={'predictions_dir': '/output.txt'}
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))

  else:
    prediction = ObjectDict({
        'outputs': {
          'predictions_dir': output_template
        }
    }).apply(gcp.use_gcp_secret('user-gcp-sa'))


  if start_step <= 4:
    confusion_matrix = dsl.ContainerOp(
      name='confusion_matrix',
        image='gcr.io/celerates-playground/ml-pipeline-local-confusion-matrix:latest',
        arguments=[
          '--predictions', prediction.output,
          '--target_lambda', lambda_target,
          '--output_dir', output_template],
      file_outputs={
        'output_dir': '/mlpipeline-metrics.json',
      }
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))
  else:
    confusion_matrix = ObjectDict({
      'outputs': {
        'output_dir': output_template
      }
    }).apply(gcp.use_gcp_secret('user-gcp-sa'))


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(kubeflow_training, __file__ + '.zip')