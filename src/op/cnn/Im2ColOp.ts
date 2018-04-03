import Tensor from "../../Tensor";
import ShapeUtils from "../../utils/ShapeUtils";
import Operation from "../Operation";

export interface Im2ColOptions {
  kernelChannel: number,
  kernelHeight: number,
  kernelNum: number,
  kernelWidth: number,
  padHeight?: number,
  padWidth?: number,
  strideHeight?: number
  strideWidth?: number,
}

export default class Im2ColOp extends Operation {

  private readonly _input: Tensor;
  private readonly _options: Im2ColOptions;
  private readonly _result: Tensor;

  get input() {
    return this._input;
  }

  get isSpecial() {
    return true;
  }

  get options() {
    return this._options;
  }

  get result() {
    return this._result;
  }

  constructor(input: Tensor, result: Tensor, options: Im2ColOptions) {
    super([input], [result]);
    this._options = options;
    this._input = input;
    this._result = result;
    if (input.rank !== 4) {
      throw new Error("image's rank is not 4");
    }
    if (input.shape[1] !== options.kernelChannel) {
      throw new Error("image channels (shape[1]) must equal kernel channels (shape[1])");
    }
  }

  exec(dim?: number): void {

    let imageNum = this.input.shape[0];
    let imageChannel = this.input.shape[1];
    let imageHeight = this.input.shape[2]; // rows
    let imageWidth = this.input.shape[3]; // cols

    let kernelNum = this.options.kernelNum;
    let kernelChannel = this.options.kernelChannel;
    let kernelHeight = this.options.kernelHeight; // rows
    let kernelWidth = this.options.kernelWidth; // cols

    let padHeight = this.options.padHeight || 0;
    let padWidth = this.options.padWidth || 0;
    let strideHeight = this.options.strideHeight || 1;
    let strideWidth = this.options.strideWidth || 1;

    let outputHeight = ShapeUtils.computeConvOutSize(imageHeight, kernelHeight, padHeight, strideHeight);
    let outputWidth = ShapeUtils.computeConvOutSize(imageWidth, kernelWidth, padWidth, strideWidth);

    let result = this.result;
    let resultIndex = 0;

    for (let c = 0; c < kernelChannel; c++) {
      for (let kRow = 0; kRow < kernelHeight; kRow++) {
        for (let kCol = 0; kCol < kernelWidth; kCol++) {

          for (let n = 0; n < imageNum; n++) {
            let inputRow = kRow - padHeight;
            for (let oR = 0; oR < outputHeight; oR++) {
              if (inputRow >= 0 && inputRow < imageHeight) {
                let inputCol = kCol - padWidth;
                for (let oC = 0; oC < outputWidth; oC++) {
                  if (inputCol >= 0 && inputCol < imageWidth) {
                    let value = this.input.get([n, c, inputRow, inputCol]);
                    result.data[resultIndex++] = value;
                  } else {
                    result.data[resultIndex++] = 0;
                  }
                  inputCol += strideWidth;
                }
              } else {
                for (let oC = 0; oC < outputWidth; oC++) {
                  result.data[resultIndex++] = 0;
                }
              }
              inputRow += strideHeight;
            }
          }

        }
      }
    }
  }

}