import Tensor from "../Tensor";

export interface TensorFormatOptions {
  decimalFormat?: string;
  padding?: number;
  precision?: number
  separator?: string;
}

export default class TensorFormat {

  private _options: TensorFormatOptions;

  constructor(options: TensorFormatOptions = {}) {
    options.separator = options.separator || '  ';
    options.padding = options.padding || 0;
    options.precision = options.precision || 8;
    this._options = options;
  }

  get separator() {
    return this._options.separator;
  }

  set separator(value) {
    this._options.separator = value;
  }

  format(tensor: Tensor): string {
    return this.formatRecursive(tensor, tensor.rank);
  }

  private formatNumber(number: number) {
    return number.toLocaleString('en-US', {
      useGrouping: false,
      maximumFractionDigits: this._options.precision
    });
  }

  private  formatRecursive(tensor: Tensor, rank: number, offset = 0): string {
    if (tensor.isScalar) {
      return this.formatNumber(tensor.get([]));
    }

    if (tensor.isVector) {
      let result = '[';
      for (let i = 0; i < tensor.length; i++) {
        result += this.formatNumber(tensor.get([i]));
        if (i < tensor.length - 1) {
          result += this._options.separator;
        }
      }
      result += ']';
      return result;
    }

    offset++;
    let result = '[';
    for (let i = 0; i < tensor.slices; i++) {
      let slice = tensor.slice(i);
      result += this.formatRecursive(slice, rank - 1, offset);
      if (i !== tensor.slices - 1) {
        result += this._options.separator + '\n';
        result += '\n'.repeat(rank - 2);
        result += ' '.repeat(offset);
      }
    }
    result += ']';
    return result;
  }
}
