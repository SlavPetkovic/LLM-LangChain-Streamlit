import os
import sys
import subprocess
import json

# Update pip package
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])

# Install a set of libraries commonly used
packages = ["pandas", "jupyter", "numpy", "logging"]
# Implement pip as a subprocess:
for element in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', element])

def create_directory_if_not_exists(directory_name):
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
        print(f"The {directory_name} directory is created!")
    else:
        print(f"{directory_name} directory already exists!")

def create_file_if_not_exists(file_path, content=""):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write(content)
        print(f"The {file_path} file is created!")
    else:
        print(f"{file_path} already exists!")

def create_config_file():
    config_data = {
        "MySQL": {
            "host": "HostName",
            "username": "username",
            "password": "password",
            "database": "database"
        },
        "SQLServerDatabase": {
            "Prod": {
                "driver": "ODBC Driver 17 for SQL Server",
                "server": "server",
                "database": "database",
                "ADUser": "Yes",
                "username": "username",
                "password": "password"
            },
            "Dev": {
                "driver": "ODBC Driver 17 for SQL Server",
                "server": "server",
                "database": "database-DEV",
                "ADUser": "Yes",
                "username": "username",
                "password": "password"
            }
        },
        "MYSQL With SSH": {
            "mypkeypass": "passkey pass",
            "sql_hostname": "hostname",
            "sql_username": "username",
            "sql_password": "password",
            "sql_main_database": "database",
            "sql_port": 3306,
            "ssh_host": "ssh host",
            "ssh_user": "ssh user",
            "ssh_port": 22,
            "ssh_password": "ssh password"
        },
        "env": "prod",
        "Email": {
            "sender_email": "do-not-reply@something.com",
            "receiver_email": "email",
            "smtp_server": "",
            "smtp_port": "25"
        },
        "OpenAI_API": "your API Key",
        "Pinecone_API": "your API Key"
    }

    config_dir = os.path.abspath("parameters")
    create_directory_if_not_exists(config_dir)
    config_file_path = os.path.join(config_dir, "config.json")

    with open(config_file_path, 'w') as config_file:
        json.dump(config_data, config_file, indent=4)

    print(f"The config.json file is created in the parameters directory!")

    # Create a templates file
    template_file_path = os.path.join(config_dir, "config_template.json")
    with open(template_file_path, 'w') as template_file:
        json.dump(config_data, template_file, indent=4)

    print(f"The config_template.json file is created in the parameters directory!")

def create_logs_directory_and_files():
    logs_dir = os.path.abspath("logs")
    create_directory_if_not_exists(logs_dir)
    log_file_path = os.path.join(logs_dir, "Process.log")
    create_file_if_not_exists(log_file_path)
    print(f"The Process.log file is created in the logs directory!")

    config_dir = os.path.abspath("parameters")
    create_directory_if_not_exists(config_dir)
    logs_ini_path = os.path.join(config_dir, "logs.ini")
    logs_ini_content = """
[loggers]
keys=root

[handlers]
keys=fileHandler

[formatters]
keys=simpleFormatter

[logger_root]
level=DEBUG
handlers=fileHandler

[handler_fileHandler]
class=FileHandler
level=DEBUG
formatter=simpleFormatter
args=("../logs/Process.log",'w')

[formatter_simpleFormatter]
format=%(asctime)s %(name)s - %(levelname)s:%(message)s
    """
    create_file_if_not_exists(logs_ini_path, logs_ini_content)
    print(f"The logs.ini file is created in the parameters directory!")

def setup_project_directories():
    create_directory_if_not_exists(os.path.abspath("data"))
    create_directory_if_not_exists(os.path.abspath("notebooks"))
    create_directory_if_not_exists(os.path.abspath("models"))
    create_directory_if_not_exists(os.path.abspath("lib"))
    create_directory_if_not_exists(os.path.abspath("parameters"))
    create_file_if_not_exists(os.path.abspath("notebooks/OpenAIAssistant.ipynb"))
    create_file_if_not_exists(os.path.abspath("README.md"))
    create_directory_if_not_exists(os.path.abspath("../assets"))
    create_config_file()
    create_logs_directory_and_files()


if __name__ == '__main__':
    setup_project_directories()