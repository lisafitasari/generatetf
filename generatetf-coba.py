import kfp
from kfp import components
from kfp import dsl
from kfp import gcp

# dataflow_tf_transform_op = 'gcr.io/celerates-playground/dock-img@sha256:e8935c4e00ce2a6d938b8349e8d03c59aadfbeca3d1022936641708494b89a92'
# kubeflow_tf_training_op  = components.load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/0b07e456b1f319d8b7a7301274f55c00fda9f537/components/kubeflow/dnntrainer/component.yaml')
# dataflow_tf_predict_op   = components.load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/0b07e456b1f319d8b7a7301274f55c00fda9f537/components/dataflow/predict/component.yaml')
# confusion_matrix_op      = components.load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/0b07e456b1f319d8b7a7301274f55c00fda9f537/components/local/confusion_matrix/component.yaml')

@dsl.pipeline(
    name='GenerateTF',
    description=''
)
def kubeflow_training(output, project,
    test='gs://dataset-image-train/TFRecords/images/test_labels.csv',
    train='gs://dataset-image-train/TFRecords/images/train_labels.csv',
    # schema='gs://ml-pipeline-playground/flower/schema.json',
    learning_rate=0.1,
    hidden_layer_size='100,50',
    steps=2000,
    target='label',
    workers=0,
    pss=0,
    preprocess_mode='local',
    predict_mode='local',
):
    output_template = str(output) + '/{{workflow.uid}}/{{pod.name}}/data'

    # set the flag to use GPU trainer
    use_gpu = False

    preprocess = dataflow_tf_transform_op(
        training_data_file_pattern=train,
        evaluation_data_file_pattern=test,
        #schema=schema,
        gcp_project=project,
        run_mode=preprocess_mode,
        preprocessing_module='',
        transformed_data_dir=output_template
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(kubeflow_training, __file__ + '.zip')
