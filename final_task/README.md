# Prompt Defender

Prompt Defender — это микросервис для классификации входящих промптов на безопасные (Regular) и потенциально опасные (Jailbreak) с использованием дообученной модели BERT. Приложение предоставляет веб-интерфейс на базе Gradio, что позволяет пользователям легко проверять текстовые запросы.

## 🔧Особенности

- **🤖Модель ML:** Используется дообученная модель BERT для классификации текстов;
- **💼Контейнеризация:** Проект упакован в Docker для удобства развертывания и масштабирования;
- **🛠️CI/CD**: программная система Jenkins для создания pipeline через Groovy;
- **🔄Версионирование данных с DVC**: система управления версиями для контроля изменений в данных.

## 🗂️Структура проекта

Проект организован в следующем виде (подкаталог `final_task`):

```
final_task/
├── my_bert_model        # Веса модели 
├── Dockerfile           # Docker-контейнер
├── .dvc                 # DVC конфигурация 
├── tests                # Папка с тестом на качество
├── datasets.dvc         # Датасет с удаленного хранилища
├── Jenkinsfile          # CI/CD     
├── requirements.txt     # Зависимости
├── main.py              # Запуск Fast API
└── prompt_defender.py   # Модуль с классификатором
```

## 🚀Настройка и запуск

1. **Клонирование репозитория:**

   ```bash
   git clone https://github.com/alex01shappard/Labs_for_MLops.git 
   cd Labs_for_MLops/final_task
   ```

2. **Установка зависимостей:**

   ```bash
   pip install -r requirements.txt
   ```

3. 🔐**Настройка удаленного хранилища:**
   - Перейдите в Google Cloud Console: 👉[https://console.cloud.google.com/](https://console.cloud.google.com/)
   - Нажмите `Select a project` → `New Project` (если нужно)
   - Перейдите: `API & Services` > `Dashboard`
   - Нажмите `Enable APIs and Services` > Найдите `Google Drive API`, включите его
   - Перейдите в меню: `API & Services` → `Credentials`
   - Нажмите `Create Credentials` → `OAuth client ID` и выберите `Desktop app`, потом нажмите `Create` и сохраните JSON-файл с ключами у себя на компьютере
   - Перейдите на Google Drive и создайте папку, в которой будут храниться данные
   - Настройте доступ для этой папки: `Общий доступ`→ `Все, у кого есть ссылка` → `Редактор`
   - Перейдите в поле URL и скопируйте последнюю часть ссылки на свое хранилище: [https://drive.google.com/drive/folders/то_что_нужно_копировать](https://drive.google.com/drive/folders/то_что_нужно_копировать)
   - Далее перейдите в терминал и пропишите следующие bash-команды:
      ```bash
      # Инициализация
      dvc init

      # Добавление remote
      dvc remote add -d myremote gdrive://то_что_скопировали_вставляем

      # Настройка клиента (если используется свой)
      dvc remote modify myremote gdrive_acknowledge_abuse true
      dvc remote modify myremote gdrive_client_id "<ваш_client_id>"
      dvc remote modify myremote gdrive_client_secret "<ваш_client_secret>"

      # Добавление данных
      dvc add data/

      # Отправка данных
      dvc push
      ```
   - `google_client_secret` сохранен в `.dvc/config.local` и исключен из версионирования git (так как это конфиденциальная информация)

4. **Запуск сервера API и первый запрос:**
   - Для запуска сервера выполните следующую команду:
     ```bash
      uvicorn main:app --host 0.0.0.0 --port 8000
     ```
   - После успешного запуска сервера API в терминале появится следующее сообщение:
     ```bash
     Loading checkpoint shards: 100%| 6/6 [00:00<00:00,  7.60it/s]
     INFO:     Started server process [6260]
     INFO:     Waiting for application startup.
     INFO:     Application startup complete.
     INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
     ```
   - Перейдите по ссылке, предоставленной в терминале, после перехода по ссылке вы увидите следующее сообщение, которое явно описывает последующий шаг работы с API:
     ```
     {"message":"Welcome to the FastAPI app! Use the /check_prompt endpoint."}
     ```
   - Перейдите в поле URL и вставьте следующую ссылку: [http://localhost:8000/check_prompt/?prompt_input="сюда_введите_свой_запрос"](http://localhost:8000/check_prompt/?prompt_input="сюда_введите_свой_запрос"), где `prompt_input` — это строка, которую вы хотите отправить на сервер API;
   - После отправки запроса вы увидите следующее сообщение:
     ```
     {"result":0,"success":true}
     ```
     где `result` — это ответ сервера API, который может меняться в зависимости от того какой запрос: безопасный (regular) запрос - `0` или вредоносный (jailbreak) - `1`, а `success` — это флаг, указывающий на то, что запрос был отправлен успешно.
   - Чтобы зыкрыть сервер нажмите сочетание клавиш `Ctrl+C`в терминале, появится следующее сообщение: 
      ```bash
      INFO:     Shutting down
      INFO:     Waiting for application shutdown.
      INFO:     Application shutdown complete.
      INFO:     Finished server process [14384]
      ```
5. **Сборка и запуск Docker-контейнера:**
   - Убедитесь, что у вас установлен Docker: [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)

   - Соберите Docker с помощью следующей команды:
      ```bash
      # Сборка контейнера
      docker build -t my-fastapi-app .
      ```
   - Запустите контейнер с помощью следующей команды:
      ```bash
      # Запуск контейнера
      docker run -d -p 8000:8000 my-fastapi-app
      ```
   - У вас запустится сервер и чтобы проверить, что контейнер запущен, то перейдите по следующей ссылке: [http://localhost:8000/](http://localhost:8000/), которая приведет к тому, что описано в пункте 4 про API.
   - Или проще сделайте проще, введите следующую bash-команду, которая выведет список запущенных контейнеров:
      ```bash
      # Просмотр запущенных контейнеров
      docker ps
      ```

6. **Настройка и запуск CI/CD(Jenkins):**

   - Загрузите GPG-ключ репозитория Jenkins:
   ```bash
   sudo wget -O /usr/share/keyrings/jenkins-keyring.asc https://pkg.jenkins.io/debian/jenkins.io-2023.key
   ```
   - Добавьте репозиторий Jenkins в источники пакетов:
   ```bash 
   echo "deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] https://pkg.jenkins.io/debian binary/" | sudo tee /etc/apt/sources.list.d/jenkins.list > /dev/null
   ```
   - Обновите список пакетов:
   ```bash
   sudo apt-get update
   ```
   - Установите Jenkins:
   ```bash
   sudo apt-get install jenkins
   ```
   - Запустите службу Jenkins при загрузке (необязательно):
   ```bash
   sudo systemctl enable jenkins
   ```
   - Запустите службу Jenkins:
   ```bash
   sudo systemctl start jenkins
   ```
   - Проверьте состояние службы Jenkins:
   ```bash
   sudo systemctl status jenkins
   ```
   Если сработало всё хорошо, то высветится в консоли следующий подобный статус: 
   ```bash
   Loaded: loaded (/lib/systemd/system/jenkins.service; enabled; vendor preset: enabled)
   Active: active (running) since Tue 2018-11-13 16:19:01 +03; 4min 57s ago
   ```
   - Откройте веб-браузер и перейдите по адресу:[http://localhost:8080/](http://localhost:8080/)
   - Выполните разблокировку Jenkins:
      Высветится инструкция, которой необходимо следовать для активации,ссылка на пароль администратора и поле ввода самого пароля `Administrator password`. После проделанной инструкции вставляем пароль в поле и идем дальше. Команда для получения пароля:
     ```bash
     sudo cat /var/lib/jenkins/secrets/initialAdminPassword
     ```
   - Установите и настройте плагин `Docker Pipeline plugin`:
      * Нажмите справа сверху `Настроить Jenkins`;
      * Потом `Plugins`→`Available plugins`;
      * Установите плагин, вернитесь на два уровня выше и нажмите `Tools`;
      * Спуститесь до раздела `Установки Docker`;
      * Задайте любое имя `Name` и в `Installation root` укажите папку, в которой находится ваш Docker, например, `/usr/bin`.

   - Выполните настройку плагинов и сам Jenkins следующим образом:
      * После нажатия кнопки продолжить, установите необходимые плагины для работы с Jenkins, где рекомендуется выбирать все плагины.
      * Далее создайте пользователя-администратора, то есть создайте логин и пароль, после написания сохраните `Save and Finish`.
      * Откройте `Dashboard` и создайте новый `item`, заходите в настройки и выбирайте раздел `Pipeline`.
      * 🔐Создание и настройка Service Account в Google Cloud Console:
         - Перейдите в свой нынешний проект и нажмите `IAM & Admin` → `Service accounts`;
         - Нажмите `Create Service Account`;
         - Введите имя, ID генерируется автоматически и введите любое описание;
         - Нажмите `Create`;
         - Потом перейдите на вкладку `Keys`;
         - Нажмите `Add key` и выберите `Create new key`, тип ключа выберите JSON;
         - Скачайте JSON-файл - он содержит секретные данные для аутентификации;
         - Откройте Google Drive, в котром находится папка с удаленным хранилищем;
         - Скопируйте из JSON-файла email-адрес (`client-email`) и добавьте нового пользователя с этим email-адресом в папку с удаленным хранилищем;
         - Установите права доступа `Редактор`
      * Далее установите следующие настройки:
         - Definition:`Pipeline script from SCM`.
         - SCM: `Git`.
         - Repository URL: `https://github.com/alex01shappard/Labs_for_MLops.git`.
         - Credentials: `- none -`:
            Добавьте кред для работы удаленного хранилища для Jenkins:
            1. Kind: `Secret file`
            2. Upload JSON-файл полученный ранее от сервисного аккаунта Google Cloud
            3. ID: `GDRIVE_SERVICE_ACCOUNT_JSON`
            4. Scope: `Global`
         - Branch Specifier (blank for 'any'): `*/jenkins`.
         - Script Path: `final_task/Jenkinsfile`
   - ⚠️Перед запуском пайплайна Jenkins запустите приложение Docker Desktop для запуска Docker Engine, иначе у вас не выполнится этап сборки Docker-образа в Jenkins!
         
## 📋**Примечания:**
Здесь можно посмотреть подробные инструкции по настройке [jenkins](https://www.jenkins.io/doc/book/installing/linux/), [DVC](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive).



