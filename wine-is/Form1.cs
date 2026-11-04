namespace CodingBackProp
{
    public partial class Form1 : Form
    {
        // does the actual work for the buttons
        private WineRunner? runner;

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object? sender, EventArgs e)
        {
            // everything printed with Console.WriteLine will show in the text box
            Console.SetOut(new TextBoxWriter(textBoxOutput));
            try
            {
                runner = new WineRunner();
            }
            catch (Exception ex)
            {
                MessageBox.Show("Could not load the data:\n" + ex.Message, "Error");
                SetBusy(true);
            }
        }

        private async void buttonTrain_Click(object? sender, EventArgs e)
        {
            if (runner == null) return;
            WineRunner r = runner;
            await RunAsync(() => r.RunChosen());
        }

        private async void buttonTune_Click(object? sender, EventArgs e)
        {
            if (runner == null) return;
            WineRunner r = runner;
            await RunAsync(() => r.RunTuning());
        }

        private async void buttonFinal_Click(object? sender, EventArgs e)
        {
            if (runner == null) return;
            WineRunner r = runner;
            // ask first, because the final test should only be done once
            string message = "This trains a final model and evaluates it on the TEST set.\n" +
              "Do it only once, after tuning is finished.";
            if (r.ModelExists)
                message += "\n\nA saved model already exists and will be OVERWRITTEN.";
            message += "\n\nContinue?";
            if (MessageBox.Show(message, "Final test", MessageBoxButtons.YesNo,
              MessageBoxIcon.Warning) != DialogResult.Yes)
                return;
            await RunAsync(() => r.RunFinal());
        }

        private void buttonClear_Click(object? sender, EventArgs e)
        {
            textBoxOutput.Clear();
        }

        // runs the work in the background
        private async Task RunAsync(Action work)
        {
            SetBusy(true);
            try
            {
                await Task.Run(work);
            }
            catch (Exception ex)
            {
                Console.WriteLine("\nError: " + ex.Message);
            }
            finally
            {
                SetBusy(false);
            }
        }

        // turns the buttons off while something is running
        private void SetBusy(bool busy)
        {
            buttonTrain.Enabled = !busy;
            buttonTune.Enabled = !busy;
            buttonFinal.Enabled = !busy;
            Cursor = busy ? Cursors.WaitCursor : Cursors.Default;
        }
    }
}