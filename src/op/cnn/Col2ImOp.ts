import Tensor from "../../Tensor";
import ShapeUtils from "../../utils/ShapeUtils";
import Operation from "../Operation";

export interface Col2ImOptions {
  imageChannel: number,
  imageHeight: number,
  imageNum: number,
  imageWidth: number,
  kernelHeight: number,
  kernelWidth: number,
  padHeight?: number,
  padWidth?: number,
  strideHeight?: number
  strideWidth?: number,
}

export default class Col2ImOp extends Operation {

  private _options: Col2ImOptions;

  constructor(input: Tensor, other: Tensor, result: Tensor, options: Col2ImOptions) {
    super(input, other, result);
    this._options = options;
    if (input.rank !== 2) {
      throw new Error("col's rank is not 2");
    }
  }

  get isSpecial() {
    return true;
  }

  get options() {
    return this._options;
  }

  exec(dim?: number): void {

    let imageNum = this.options.imageNum;
    let imageChannel = this.options.imageChannel;
    let imageHeight = this.options.imageHeight; // rows
    let imageWidth = this.options.imageWidth; // cols

    let kernelHeight = this.options.kernelHeight; // rows
    let kernelWidth = this.options.kernelWidth; // cols

    let padHeight = this.options.padHeight || 0;
    let padWidth = this.options.padWidth || 0;
    let strideHeight = this.options.strideHeight || 1;
    let strideWidth = this.options.strideWidth || 1;

    let outputHeight = ShapeUtils.computeConvOutSize(imageHeight, kernelHeight, padHeight, strideHeight);
    let outputWidth = ShapeUtils.computeConvOutSize(imageWidth, kernelWidth, padWidth, strideWidth);

    let result = this.result;
    let col = this.input;

    let dataIndex = 0;

    for (let c = 0; c < imageChannel; c++) {
      for (let kRow = 0; kRow < kernelHeight; kRow++) {
        for (let kCol = 0; kCol < kernelWidth; kCol++) {

          for (let n = 0; n < imageNum; n++) {

            let inputRow = kRow - padHeight;
            for (let oR = 0; oR < outputHeight; oR++) {

              if (inputRow < 0 || inputRow >= imageHeight) {
                dataIndex += outputWidth;
                continue;
              }

              let inputCol = kCol - padWidth;
              for (let oC = 0; oC < outputWidth; oC++) {
                if (inputCol >= 0 && inputCol < imageWidth) {
                  let colIndex = n * imageChannel * imageWidth * imageHeight + c * imageWidth * imageHeight + inputRow * imageWidth + inputCol;
                  result.data[colIndex] += col.data[dataIndex];
                }
                dataIndex++;
                inputCol += strideWidth;
              }

              inputRow += strideHeight;
            }
          }

        }
      }
    }

  }

}