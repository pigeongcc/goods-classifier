# Классификатор товаров по категориям КТРУ
Проект в рамках стажировки в Институте ИИ Университета Иннополис. Октябрь 2022

## Задача
Дан датасет товаров со множеством признаков:
- Категориальные признаки: несколько категорий товара по классификациям, принятым в РФ, и категория компании-проиводителя.
Классификации иерархические: доступны названия категорий и их родителей.
- Текстовые признаки: название товара, описание, ключевые слова, название компании, наименования всех категорий из категориальных признаков.

Необходимо предсказать категорию товара по КТРУ (ещё одна классификация товаров).

Датасет содержит >300k размеченных товаров и >700k неразмеченных.

## Решение
- **Расширение** датасета через открытые источники (+40k размеченных товаров)
- **Разметка** датасета вручную (+7k размеченных товаров)
- **Вывод признаков**
  - получение цепочки категорий - от листовой до корневой - для каждой доступной классификации. Этот шаг дал сильный прирост метрик
  - добавление признаков компании-производителя
  - аггрегация всех доступных признаков в одной таблице
- **Анализ данных** показал сильный дисбаланс таргета в размеченной части датасета. Для тестирования модели решил использовать две версии датасета:
  - Сбалансированный - баланс классов соблюдён
  - Стратифицированный - баланс классов воспроизводит исходный датасет
- **Построение модели** (model.py):
  - Градиентный бустинг на решающих деревьях (catboost)
  - Подбор гиперпараметров (Optuna)
- **Инференс**
  - Обернул модель и датасет в классы (model.py, dataset_builder.py)
  - **Функционал классов и пример работы - в ноутбуке run.ipynb**
  
- Итоговая метрика **F1 Macro score: 0.99**
