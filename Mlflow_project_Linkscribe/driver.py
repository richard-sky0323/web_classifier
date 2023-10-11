import mlflow

# mlflow run https://github.com/haruiz/iris-project -P nsplits=20
if __name__ == '__main__':
    mlflow.projects.run(
        'https://github.com/richard-sky0323/web_classifier',
        backend='local',
        parameters={
            'nsplits': 5
        })
    
    