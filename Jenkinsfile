pipeline {
    agent any

    environment {
        BACKEND_IMAGE = 'viodetect-backend-image'
        FRONTEND_IMAGE = 'viodetect-frontend-image'
        NETWORK_NAME = 'viodetect_mynetwork'
        BACKEND_CONTAINER = 'viodetect-backend'
        FRONTEND_CONTAINER = 'viodetect-frontend'
    }

    stages {
        stage('Clone Repo') {
            steps {
                git 'https://github.com/harmeshgv/YoloV8-LSTM-video-Classification.git'
            }
        }

        stage('Build Docker Images') {
            steps {
                script {
                    powershell """
                    docker compose -f docker-compose.yml build
                    """
                }
            }
        }

        stage('Run Containers') {
            steps {
                script {
                    powershell """
                    docker network create ${NETWORK_NAME}
                    docker run -d --rm --name ${BACKEND_CONTAINER} --network ${NETWORK_NAME} -p 8000:8000 ${BACKEND_IMAGE}
                    docker run -d --rm --name ${FRONTEND_CONTAINER} --network ${NETWORK_NAME} -p 8501:8501 ${FRONTEND_IMAGE}
                    """
                }
            }
        }

        stage('Check Backend Health') {
            steps {
                script {
                    sleep 10
                    def response = powershell(returnStdout: true, script: '''
                    try {
                        $status = (Invoke-WebRequest -Uri http://localhost:8000/docs -UseBasicParsing -TimeoutSec 10).StatusCode
                        Write-Output $status
                    } catch {
                        Write-Output "ERROR"
                    }
                    ''').trim()

                    if (response != '200') {
                        error "Backend health check failed. HTTP Status: ${response}"
                    }
                }
            }
        }
    }

    post {
        always {
            echo 'Cleaning up containers...'
            powershell """
            docker stop ${FRONTEND_CONTAINER}
            docker stop ${BACKEND_CONTAINER}
            docker network rm ${NETWORK_NAME}
            """
        }
    }
}
