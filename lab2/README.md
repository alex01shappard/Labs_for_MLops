# Лабораторная работа №2. Создание Jenkins Pipeline для автоматизации конвейера машинного обучения 

Этот проект демонстрирует создание автоматизированного конвейера машинного обучения (ML) с помощью Jenkins, интегрируя этапы сбора данных, обучения модели, тестирования и деплоя. Основная цель — автоматизировать жизненный цикл ML-модели, обеспечивая воспроизводимость, масштабируемость и контроль качества на всех этапах.

## Особенности
 * **Модель ML**: Используется обученная модель линейной регрессии из первой лаб. работы: `Labs_for_MLops/lab1/model.pkl`.
 * **CI/CD**: программная система Jenkins для создания pipeline через Groovy.

## Этапы конвейера:
1. **Создание данных и признаков (`data_creation.py`)**:
    - Загрузка данных из папок `lab1/train` и `lab1/test`.
    - Генерация признаков на основе дня года (синус-косинусные преобразования).
    - Сохранение признаков (`X_train, x_test`) и целевой переменной (`y_train, y_test`) в отдельные CSV-файлы, которые располагаются в папках `lab2/train` и `lab2/test`.
2. **Обучение/переобучение модели (`model_training.py`)**:
    - Загрузка тренировочных данных (`X_train, y_train`) из папки `lab2/train`.
    - Проверка существования модели в `lab1`.
    - Сохранение параметра периода `T` вместе с моделью для воспроизводимости в `model.pkl`.
    - Поддержка создания и обучения новой модели и дообучение существующей.
3. **Тестирование модели (`model_testing.py`)**:
    - Загрузка тренировочных данных (`X_test, y_test`) из папки `lab2/test`.
    - Загрузка обученной модели и параметра `T` из файла `lab2/model.pkl`.
    - Расчет R²-метрики для оценки предсказательной способности модели.

## Структура проекта:

```
.
└── lab2
    ├── scripts
    |   ├── data_processing.py
    |   ├── model_testing.py
    |   └── model_training.py
    ├── test
    |   ├── X_test.csv
    |   └── y_test.csv
    ├── train
    |   ├── X_train.csv
    |   └── y_train.csv
    ├── Jenkinsfile
    ├── model.pkl
    ├── README.md
    └── requirements.txt
```

## Установка и запуск
1. **Загрузка GPG-ключа репозитория Jenkins:**
```bash
sudo wget -O /usr/share/keyrings/jenkins-keyring.asc https://pkg.jenkins.io/debian/jenkins.io-2023.key
```
2. **Добавление репозитория Jenkins в источники пакетов:**
```bash
echo "deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] https://pkg.jenkins.io/debian binary/" | sudo tee /etc/apt/sources.list.d/jenkins.list > /dev/null
```
3. **Обновление списка пакетов:**
```bash
sudo apt-get ipdate
```
4. **Установка Jenkins:**
```bash
sudo apt-get install jenkins
```
5. **Запуск службы Jenkins при загрузке:**
```bash
sudo systemctl enable jenkins
```
6. **Запуск службы Jenkins:**
```bash
sudo systemctl start jenkins
```
7. **Проверка состояния службы Jenkins:**
```bash
sudo systemctl status jenkins
```
Если сработало всё хорошо, то высветится в консоли следующий статус: 
```bash
Loaded: loaded (/lib/systemd/system/jenkins.service; enabled; vendor preset: enabled)
Active: active (running) since Tue 2018-11-13 16:19:01 +03; 4min 57s ago
```
8. **Открытие запустившегося сервера:**
[http://localhost:8080](http://localhost:8080)

9. **Разблокировка Jenkins:**
Высветится инструкция, которой необходимо следовать для активации, ссылка на пароль администратора и поле ввода самого пароля `Administrator password`. После проделанной инструкции вставляем пароль в поле и идем дальше. Команда для получения пароля:
```bash
sudo cat /var/lib/jenkins/secrets/initialAdminPassword
```
10. **Установка плагинов и настройка Jenkins:**
    * После нажатия кнопки продолжить, предлагается установить необходимые плагины для работы с Jenkins, где рекомендуется выбирать все плагины.
    * Далее предлагается создание пользователя-администратора, то есть создание логина и пароля, после написания сохраняем `Save and Finish`.
    * Открываем `Dashboard` и создаем новый `item`, заходим в настройки и выбираем раздел `Pipeline`.
    * Далее устанавливаем следующие настройки:
      - Definition:`Pipeline script from SCM`.
      - SCM: `Git`.
      - Repository URL: `свой_url_репозитория.git`.
      - Credentials: `- none -`.
      - Branch Specifier (blank for 'any'): `*/main`.
      - Script Path: `вводится_путь_где_лежит_скрипт_Jenkins/Jenkinsfile`

## **Примечания:**
Здесь можно посмотреть подробную [инструкцию](https://www.jenkins.io/doc/book/installing/linux/) по настройке Jenkins