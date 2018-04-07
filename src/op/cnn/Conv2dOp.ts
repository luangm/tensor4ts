import Tensor from "../../Tensor";
import ShapeUtils from "../../utils/ShapeUtils";
import Operation from "../Operation";
import TensorMath from "../../TensorMath";

export interface Conv2dOptions {
  padHeight?: number,
  padWidth?: number,
  strideHeight?: number
  strideWidth?: number,
}

export default class Conv2dOp extends Operation {

  private readonly _image: Tensor;
  private readonly _kernel: Tensor;
  private readonly _options: Conv2dOptions;
  private _result: Tensor;

  get image() {
    return this._image;
  }

  get isSpecial() {
    return true;
  }

  get kernel() {
    return this._kernel;
  }

  get options() {
    return this._options;
  }

  get result() {
    return this._result;
  }

  constructor(image: Tensor, kernel: Tensor, options: Conv2dOptions, result: Tensor) {
    super([image, kernel], [result]);
    this._options = options;
    this._image = image;
    this._kernel = kernel;
    this._result = result;
    if (image.rank !== 4) {
      throw new Error("image's rank must be 4: NxCxHxW");
    }
    if (kernel.rank !== 4) {
      throw new Error("kernel's rank must be 4: NxCxHxW");
    }
  }

  exec(dim?: number): void {
    let imageShape = this.image.shape;
    let kernelShape = this.kernel.shape;
    let outputShape = this.result.shape;
    let xCol = TensorMath.im2col(this.image, {
      kernelNum: kernelShape[0],
      kernelChannel: kernelShape[1],
      kernelHeight: kernelShape[2],
      kernelWidth: kernelShape[3],
      padHeight: this.options.padHeight,
      padWidth: this.options.padWidth,
      strideHeight: this.options.strideHeight,
      strideWidth: this.options.strideWidth
    });

    let kRows = this.kernel.reshape([kernelShape[0],-1]);
    let result = TensorMath.matmul(kRows, xCol);
    let reshaped = result.reshape([kernelShape[0], imageShape[0], outputShape[2], outputShape[3]]);
    let transposed = reshaped.transpose([1, 0, 2, 3]);
    this._result = transposed;
  }

}