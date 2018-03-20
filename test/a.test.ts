import {Tensor} from "../src/index";
import TensorMath from "../src/TensorMath";

test("A", function () {
    let tensor = Tensor.create(1);
    console.log(tensor);

    let t2 = Tensor.create([[1, 2, 3], [4, 5, 6]]);
    console.log(t2);

    let sum = TensorMath.add(tensor, t2);
    console.log(sum);
});