# Car Damage Assessment - A proof of concept

## Use Case:
To reduce claim processing in auto insurance industry by automating the car damage assessment. In general industry the claims are filed manually after which inspectors are required to physcially look over vehicles and make damage assessments. By automating the process, claims can be filed whenever and wherever and estiamtes can be reached faster. Inspectors can now be allowed to focus on more complex and larger claims that are yet too difficult to be virtually processed. This increases both effiiceny and productivity.

While the technology is yet to achieve the highest possible levels of accuracy, above is a proof of concept of the appication of Deep Learning and Computer Vision into automating the damage assessments by building and training Convolution Neural Networks.

## Solution:
The model accepts an input image from the user and processes it across 4 stages:
1. Validates that given image is of a car.
2. Validates that the car is damaged.
3. Finds location of damage as front, rear or side
4. Determines severity of damage as minor, moderate or severe

The model can also further be imporved to:
1. Obtain a cost estimate
2. Send assessment to insurer carrier
3. Print documentation

![Alt Text](https://https://github.com/catthatcodes/car-damage-assessment/demo.gif)
