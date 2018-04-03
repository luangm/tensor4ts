import {Tensor, TensorMath} from "../src/index";

test("conditional", function () {

  let condition = Tensor.create([[1, 1, 0], [1, 0, 1]]);
  let truthy = Tensor.create([[1, 2, 3], [4, 5, 6]]);
  let falsy = Tensor.create([[2, 5, 8], [3, 6, 9]]);

  let z = TensorMath.conditional(condition, truthy, falsy);
  let expected = Tensor.create([[1, 2, 8], [4, 6, 6]]);
  expect(z).toEqual(expected);
});
