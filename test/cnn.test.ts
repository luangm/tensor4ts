import {Tensor, TensorMath} from "../src/index";

test("im2col", function () {
  let image = Tensor.linspace(1, 16, 16).reshape([1, 1, 4, 4]);
  let result = TensorMath.im2col(image, {
    kernelWidth: 2,
    kernelHeight: 2,
    kernelNum: 1,
    kernelChannel: 1
  });

  let expected = Tensor.create([
    [1, 2, 3, 5, 6, 7, 9, 10, 11],
    [2, 3, 4, 6, 7, 8, 10, 11, 12],
    [5, 6, 7, 9, 10, 11, 13, 14, 15],
    [6, 7, 8, 10, 11, 12, 14, 15, 16]
  ]);

  expect(result).toEqual(expected);
});

test("col2im", function () {
  let col = Tensor.create([
    [1, 2, 3, 5, 6, 7, 9, 10, 11],
    [2, 3, 4, 6, 7, 8, 10, 11, 12],
    [5, 6, 7, 9, 10, 11, 13, 14, 15],
    [6, 7, 8, 10, 11, 12, 14, 15, 16]
  ]);

  let image = TensorMath.col2im(col, {
    kernelWidth: 2,
    kernelHeight: 2,
    imageWidth: 4,
    imageHeight: 4,
    imageNum: 1,
    imageChannel: 1
  });

  let expected = Tensor.create([
    [1, 4, 6, 4],
    [10, 24, 28, 16],
    [18, 40, 44, 24],
    [13, 28, 30, 16]
  ]).reshape([1, 1, 4, 4]);

  expect(image).toEqual(expected);
});

test("im2col - col2im", function () {
  let image = Tensor.linspace(1, 16, 16).reshape([1, 1, 4, 4]);
  let col = TensorMath.im2col(image, {
    kernelWidth: 2,
    kernelHeight: 2,
    kernelNum: 1,
    kernelChannel: 1,
    strideWidth: 2,
    strideHeight: 2
  });

  // console.log(col.toString());
  //
  let image1 = TensorMath.col2im(col, {
    kernelWidth: 2,
    kernelHeight: 2,
    imageWidth: 4,
    imageHeight: 4,
    imageNum: 1,
    imageChannel: 1,
    strideWidth: 2,
    strideHeight: 2
  });

  expect(image1).toEqual(image);

  // expect(result).toEqual(expected);
});

test("conv2d", function () {
  let image = Tensor.linspace(1, 9, 9).reshape([1, 1, 3, 3]);
  let kernel = Tensor.linspace(1, 4, 4).reshape([1, 1, 2, 2]);

  let result = TensorMath.conv2d(image, kernel, {strideWidth: 1, strideHeight: 1, padWidth: 1, padHeight: 1});

  console.log(result.toString());
  // expect(result).toEqual(expected);
});

test("maxpool", function () {
  let image = Tensor.linspace(1, 16, 16).reshape([1, 1, 4, 4]);

  let result = TensorMath.maxPool(image, {
    strideWidth: 2,
    strideHeight: 2,
    kernelHeight: 2,
    kernelWidth: 2,
    kernelChannel: 1,
    kernelNum: 1
  });

  console.log(result);
});