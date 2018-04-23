import {Tensor, TensorMath} from '../src/index';
import Shape from "../src/Shape";
import FloatTensor from "../src/tensor/FloatTensor";
import ShapeUtils from "../src/utils/ShapeUtils";

test('reshape', function () {

  // let x = Tensor.create([[1, 2, 3], [4,5,6]]);
  // let z = x.reshape([-1, 6]);
  // console.log(z.toString());
  //
  // expect(z.shape).toEqual([1, 6]);
  //
  // let z2 = x.reshape([2, -1]);
  // expect(z2.shape).toEqual([2, 3]);
});


test('infer shape', function () {

  let shape = [2, 3, -1];
  let length = 6;

  let inferred = ShapeUtils.inferShape(length, shape);
  console.log(inferred);

});