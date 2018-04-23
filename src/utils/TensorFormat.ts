import Tensor from "../Tensor";

export interface TensorFormatOptions {
  decimalFormat?: string;
  padding?: number;
  precision?: number
  separator?: string;
}

export default class TensorFormat {

  private _options: TensorFormatOptions;

  get separator() {
    return this._options.separator;
  }

  set separator(value) {
    this._options.separator = value;
  }

  constructor(options: TensorFormatOptions = {}) {
    options.separator = options.separator || "  ";
    options.padding = options.padding || 0;
    options.precision = options.precision || 8;
    this._options = options;
  }

  format(tensor: Tensor): string {
    return this.formatRecursive(tensor, tensor.rank);
  }

  private formatNumber(number: number) {
    return number.toLocaleString("en-US", {
      useGrouping: false,
      maximumFractionDigits: this._options.precision
    });
  }

  private formatRecursive(tensor: Tensor, rank: number, offset = 0): string {
    if (tensor.rank === 0) {
      return this.formatNumber(tensor.get([]));
    }

    if (tensor.rank === 1) {
      let result = "[";
      for (let i = 0; i < tensor.length; i++) {
        result += this.formatNumber(tensor.get([i]));
        if (i < tensor.length - 1) {
          result += this._options.separator;
        }
      }
      result += "]";
      return result;
    }

    let slices = tensor.shape[0];
    offset++;
    let result = "[";
    for (let i = 0; i < slices; i++) {
      let slice = tensor.sliceSingle(i);
      result += this.formatRecursive(slice, rank - 1, offset);
      if (i !== slices - 1) {
        result += this._options.separator + "\n";
        result += "\n".repeat(rank - 2);
        result += " ".repeat(offset);
      }
    }
    result += "]";
    return result;
  }
}
