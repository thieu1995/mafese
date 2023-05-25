#!/usr/bin/env python
# Created by "Thieu" at 10:53, 25/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mafese import get_dataset

# Try unknown data
get_dataset("unknown")
# Enter: 1

data = get_dataset("Arrhythmia")
data.split_train_test(test_size=0.2)

print(data.X_train[:2].shape)
print(data.y_train[:2].shape)
