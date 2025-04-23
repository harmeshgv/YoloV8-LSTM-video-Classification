pipeline {
    agent any

    environment {
        COMPOSE_PROJECT_NAME = "viodetect-project"
    }

    stages {
        stage('Clone Repository') {
            steps {
                git url: 'https://github.com/harmeshgv/YoloV8-LSTM-video-Classification.git',
                    branch: 'main'
            }
        }

        stage('Docker Compose Up') {
            steps {
                bat 'docker-compose down || exit 0' // Ensure any previous containers are stopped
                bat 'docker-compose up --build -d' // Add '-d' for detached mode (run in background)
            }
        }
    }

    post {
        always {
            echo 'Pipeline started'
        }

        success {
            echo 'Pipeline finished successfully'
        }

        failure {
            echo 'Pipeline failed. Investigate the logs for errors.'
        }
    }
}
