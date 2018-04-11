import {Tensor} from "../src/index";
import TensorMath from "../src/TensorMath";

test("addN", function () {
  let x = Tensor.linspace(1, 6, 6).reshape([2, 3]);
  let y = Tensor.linspace(2, 7, 6).reshape([2, 3]);
  let z = Tensor.linspace(3, 8, 6).reshape([2, 3]);

  let result = TensorMath.addN([x, y, z]);

  let expected = Tensor.create([[1 + 2 + 3, 2 + 3 + 4, 3 + 4 + 5], [4 + 5 + 6, 5 + 6 + 7, 6 + 7 + 8]]);
  expect(result).toEqual(expected);
});