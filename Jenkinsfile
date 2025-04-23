pipeline {
    agent any

    stages {
        stage('Clone Repo') {
            steps {
                git branch: 'main',
                url: 'https://github.com/harmeshgv/YoloV8-LSTM-video-Classification.git'
            }
        }

        stage('Build Images') {
            steps {
                powershell 'docker-compose build'
            }
        }

        stage('Run Containers') {
            steps {
                powershell 'docker-compose up -d'
            }
        }
    }
}
