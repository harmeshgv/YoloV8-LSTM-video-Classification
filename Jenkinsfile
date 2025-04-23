pipeline {
    agent any

    environment {
        COMPOSE_PROJECT_NAME = "viodetect"
    }

    stages {
        stage('Clone Repo') {
            steps {
                git branch: 'main',
                    url: 'https://github.com/harmeshgv/YoloV8-LSTM-video-Classification.git'
            }
        }

        stage('Build Docker Images') {
            steps {
                script {
                    // Ensure the Docker Compose file exists
                    if (fileExists('docker-compose.yml')) {
                        powershell 'docker-compose build'
                    } else {
                        error "docker-compose.yml not found!"
                    }
                }
            }
        }

        stage('Run Containers') {
            steps {
                script {
                    powershell 'docker-compose up -d'
                }
            }
        }

        stage('Check Backend Health') {
            steps {
                script {
                    sleep(time: 10, unit: 'SECONDS') // Wait for services to start
                    def response = powershell(
                        returnStdout: true,
                        script: 'curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/docs'
                    ).trim()
                    if (response != "200") {
                        error("Backend is not responding properly. Got status: ${response}")
                    }
                }
            }
        }
    }

    post {
        always {
            echo "Cleaning up containers..."
            powershell 'docker-compose down'
        }
    }
}
