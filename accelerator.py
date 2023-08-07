from azureml.core import Workspace, Environment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import ScriptRunConfig
from azureml.core import Experiment


import logging


def gpu_conn():
    subscription_id = "e1508c0b-823c-4c5b-931f-7eafb750a697"
    tenant_id = "61210773-40ad-4849-a6f1-e9f5d243d3d4"
    client_id = "bd88f437-7d17-4b27-b76a-e8ed5d4ce62f"
    client_secret = "2hSsFhxie1XLPpa0Al~X_G9xjln7vI_g6k"

    # Connect to the Azure ML Service Workspace using a service principal
    svcpr = ServicePrincipalAuthentication(
        tenant_id=tenant_id,
        service_principal_id="bd88f437-7d17-4b27-b76a-e8ed5d4ce62f",
        service_principal_password="VXVDG7vwfD.0qmicmxpGwRWIa~LO~GZh.U")

    ws = Workspace(
        subscription_id=subscription_id,
        resource_group="Genieaz",
        workspace_name="genieaz-ml-ws",
        auth=svcpr)


    env_name = 'AzureML-pytorch-1.10-ubuntu18.04-py38-cuda11-gpu'
    genieaz_ml_core_env = Environment.get(workspace=ws, name=env_name)

    cluster_name = "genieaz-ml-cluster"

    try:
        compute_target = ComputeTarget(workspace=ws, name=cluster_name)
        print('Found existing compute target.')
        logging.info('Found existing compute target.')
    except ComputeTargetException:
        print('Creating a new compute target...')
        logging.info('Creating a new compute target...')
        compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_NC4as_T4_v3',
                                                               max_nodes=1,
                                                               vm_priority='lowpriority')

        # Create the cluster.
        compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
        compute_target.wait_for_completion(show_output=True)

    # Use get_status() to get a detailed status for the current AmlCompute.
    print(compute_target.get_status().serialize())

    src = ScriptRunConfig(source_directory='',
                          command=['/bin/bash', 'training-server/train-script.sh'],
                          compute_target=compute_target,
                          environment=genieaz_ml_core_env)

    run = Experiment(workspace=ws, name='my-experiment').submit(src)
    run.wait_for_completion(show_output=True)

    print("All Done Training the training-server")
    logging.info("All Done Training the training-server")
