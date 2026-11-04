using System.Text;

namespace CodingBackProp
{
    public class TextBoxWriter : TextWriter
    {
        private readonly TextBox box;

        public TextBoxWriter(TextBox box)
        {
            this.box = box;
        }

        public override Encoding Encoding
        {
            get { return Encoding.UTF8; }
        }

        public override void Write(char value)
        {
            if (value == '\r') return;
            Append(value == '\n' ? "\r\n" : value.ToString());
        }

        public override void Write(string? value)
        {
            if (value != null)
                Append(value.Replace("\r", "").Replace("\n", "\r\n"));
        }

        public override void Write(char[] buffer, int index, int count)
        {
            Write(new string(buffer, index, count));
        }
        private void Append(string text)
        {
            if (box.IsDisposed || !box.IsHandleCreated) return;
            if (box.InvokeRequired)
                box.BeginInvoke(new Action(() => box.AppendText(text)));
            else
                box.AppendText(text);
        }
    }
}